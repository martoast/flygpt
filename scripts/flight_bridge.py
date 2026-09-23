"""Serve a drone pilot actor to the fire-drone simulator over local HTTP.

The simulator's GoalPilot calls ``POST /act`` once per inner physics step with
the same 83-number observation it would give its in-browser ONNX actor, and
applies the returned four-channel action. ``POST /reset`` starts each flight,
so a recurrent actor never carries state between flights. The simulator is
fixed-step: bridge latency changes wall-clock time, never sim outcomes.

Backends:
  onnx  -- run an exported actor with onnxruntime (CP0 transport parity check)

Usage (from the fire-drone sim venv, which has onnxruntime):
  python scripts/flight_bridge.py --backend onnx --model <policy.onnx> --port 8799
"""
import argparse
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

OBSERVATION_SIZE = 83
ACTION_SIZE = 4


class OnnxActor:
    """Stateless exported actor; reset is a no-op."""

    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}
        with open(path, 'rb') as f:
            self.sha256 = hashlib.sha256(f.read()).hexdigest()

    def reset(self):
        pass

    def act(self, observation, clearance=None):
        feeds = {'observation': observation[None, :]}
        if clearance is not None:
            if 'clearance' not in self.inputs:
                raise ValueError('actor has no clearance input')
            feeds['clearance'] = clearance[None, :]
        return self.session.run(['action'], feeds)[0][0]


class Bridge:
    def __init__(self, actor):
        self.actor = actor
        self.lock = threading.Lock()
        self.flight = None
        self.steps = 0

    def reset(self, body):
        with self.lock:
            self.actor.reset()
            self.flight = body.get('flight')
            self.steps = 0
        return {'ok': True, 'flight': self.flight, 'modelSHA256': self.actor.sha256}

    def act(self, body):
        observation = np.asarray(body['observation'], dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError(f'observation must be {OBSERVATION_SIZE} finite floats')
        clearance = body.get('clearance')
        if clearance is not None:
            clearance = np.asarray(clearance, dtype=np.float32)
        with self.lock:
            if self.flight is None or body.get('flight') != self.flight:
                raise ValueError('act before reset, or for a different flight')
            action = np.asarray(self.actor.act(observation, clearance), dtype=np.float32)
            self.steps += 1
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
            raise ValueError('actor returned an invalid action')
        return {'action': action.tolist()}


def handler_for(bridge):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path == '/health':
                self._send(200, {'ok': True, 'modelSHA256': bridge.actor.sha256})
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                route = {'/reset': bridge.reset, '/act': bridge.act}.get(self.path)
                if route is None:
                    return self._send(404, {'error': 'not found'})
                self._send(200, route(body))
            except Exception as error:  # surfaced to the sim, which fails the flight
                self._send(400, {'error': str(error)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['onnx'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8799)
    args = parser.parse_args()
    actor = OnnxActor(args.model)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Bridge(actor)))
    print(json.dumps({'listening': args.port, 'backend': args.backend, 'modelSHA256': actor.sha256}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()

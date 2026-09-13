"""Notify the local user when saved compiler-study milestones are complete."""
import json
import os
from pathlib import Path
import subprocess
import time
from src.provenance import save_json

ROOT=Path('results/compiler_v1')

def notify(key,message):
    receipt=ROOT/'notifications'/f'{key}.json'
    if receipt.exists():return
    escaped=message.replace('\\','\\\\').replace('"','\\"')
    result=subprocess.run(['osascript','-e',f'display notification "{escaped}" with title "FlyGPT"'],capture_output=True,text=True)
    save_json(receipt,{'message':message,'notification_requested':result.returncode==0,'error':result.stderr,'time':time.time(),
        'note':'Desktop visibility depends on macOS notification settings; this is not an unsolicited chat message'})

def main():
    lock=ROOT/'notification_watcher.lock'
    if lock.exists():
        try:os.kill(json.loads(lock.read_text())['pid'],0)
        except ProcessLookupError:pass
        else:return
    save_json(lock,{'pid':os.getpid()})
    try:
        while True:
            first=ROOT/'paired_ce/seed_0/comparison.json'
            if first.exists():
                r=json.loads(first.read_text())['metrics']
                notify('rewired_ce_ready',f"Seed-zero comparison ready: MaleCNS CE {100*r['real_ce']['accuracy']:.1f}%, rewired CE {100*r['rewired_ce']['accuracy']:.1f}%. See COMPILER_REPORT.md.")
            if all((ROOT/f'paired_ce/seed_{seed}/comparison.json').exists() for seed in range(5)):
                notify('five_seed_ce_ready','All five paired MaleCNS/rewired CE comparisons are ready. See COMPILER_REPORT.md.')
            if (ROOT/'failure.json').exists():
                notify('execution_stopped','The compiler study stopped on an execution error. Check COMPILER_REPORT.md.');break
            if (ROOT/'completion.json').exists():
                notify('compiler_study_ready','The frozen compiler study is complete. Results are in COMPILER_REPORT.md.');break
            time.sleep(30)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()

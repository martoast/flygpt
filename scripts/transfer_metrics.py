"""Normalized loss-gap closure, with explicit baselines and attribution limits."""
import json
from pathlib import Path
from src.provenance import save_json


def main():
    root=Path('results/malecns_v1');pilot=json.loads((root/'language'/'real_0.json').read_text())
    rows=json.loads((root/'target'/'trajectory.json').read_text())['records'];result=[]
    for row in rows:
        teacher=row['teacher_nats_per_byte'];student=row['validation_nats_per_byte']
        values={'continuation_step':row['continuation_step'],'teacher_ce':teacher,'student_ce':student}
        for label,baseline in [('untrained',pilot['untrained']['ce']),('unigram',pilot['unigram_ce']),('supervised_pilot',pilot['validation']['ce'])]:
            denominator=baseline-teacher
            values[f'E_from_{label}']=None if denominator<=0 else (baseline-student)/denominator
        result.append(values)
    save_json(root/'target'/'transfer_metrics.json',{'formula':'E=(L0-Lstudent)/(L0-Lteacher)',
        'interpretation':'Normalized loss-gap closure, not a causal estimate of the benefit of distillation. Supervised-pilot learning is included in the untrained-baseline metric. A matched CE-only continuation is required to attribute improvements specifically to the teacher KL objective.',
        'records':result})


if __name__=='__main__':main()

"""Export the measured baseline trajectory. No smoothing or extrapolation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='flygpt-trajectory-v1'
import matplotlib.pyplot as plt


def main():
    root=Path('results/malecns_v1/target');path=root/'trajectory.json'
    if not path.exists():return
    rows=json.loads(path.read_text())['records']
    if not rows:return
    x=[r['bytes_seen']['total'] for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    ax=axes[0,0];ax.plot(x,[r['validation_nats_per_byte'] for r in rows],'o-',label='MaleCNS intact')
    ax.plot(x,[r['zero_edge_ablation_nats_per_byte'] for r in rows],'s--',label='Zero recurrent edges')
    ax.axhline(rows[0]['teacher_nats_per_byte'],color='black',linestyle=':',label='Frozen TinyGPT')
    ax.set(ylabel='Validation nats / byte',title='Prediction and causal ablation');ax.legend(fontsize=8)
    ax=axes[0,1];ax.plot(x,[r['gap_to_teacher'] for r in rows],'o-');ax.set(ylabel='Nats / byte',title='Gap to teacher')
    ax=axes[0,2];ax.plot(x,[r['teacher_kl_nats_per_byte'] for r in rows],'o-');ax.set(ylabel='KL nats / byte',title='KL(teacher || MaleCNS), T=1')
    ax=axes[1,0];ax.plot(x,[r['teacher_argmax_agreement'] for r in rows],'o-');ax.set(ylim=(0,1),ylabel='Fraction',title='Teacher argmax agreement')
    ax=axes[1,1];ax.plot(x,[r['gradient_norm']['last_unclipped'] for r in rows],'o-');ax.set(ylabel='L2 norm before clipping',title='Training gradient norm')
    ax=axes[1,2];ax.plot(x,[r['hidden_state']['mean_rms'] for r in rows],'o-',label='Hidden RMS')
    ax.plot(x,[r['hidden_state']['saturation_fraction'] for r in rows],'s--',label='Fraction |h| > .95')
    ax.set(title='Hidden state at byte boundaries');ax.legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Total training bytes seen');ax.grid(alpha=.2)
    fig.suptitle('Baseline A: full fixed MaleCNS graph, synthetic grammar, one seed\nSame 256 validation bytes; measured checkpoints only; no convergence or topology-advantage claim',fontsize=12)
    fig.savefig(root/'trajectory.png',dpi=160);fig.savefig(root/'trajectory.svg',metadata={'Date':None});plt.close(fig)


if __name__=='__main__':main()

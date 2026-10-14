"""Plot the recorded trajectory, not a hand-authored animation."""
from pathlib import Path
import argparse
import csv
import json
import os
import tempfile
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'quadruped-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder):
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    with (folder/'trajectory.csv').open(encoding='utf-8') as f:r=list(csv.DictReader(f))
    def col(k):return np.array([float(a[k]) for a in r])
    t=col('t');air=np.array([a['phase']=='flight' for a in r]);tl=t[air][0];td=t[air][-1]
    fig,axs=plt.subplots(4,1,figsize=(10,11),sharex=True,constrained_layout=True)
    axs[0].plot(t,col('z'),label='Trunk height',color='#2563eb')
    axs[0].plot(t,col('com_z'),label='Whole-robot COM height',color='#d97706')
    axs[0].axhline(summary['step_height_m'],color='#15803d',ls=':',label='Platform top')
    axs[0].set_ylabel('Height (m)');axs[0].legend(loc='upper right',ncol=2)
    axs[1].plot(t,col('com_vx'),label='COM horizontal speed',color='#2563eb')
    axs[1].plot(t,col('com_vz'),label='COM vertical speed',color='#d97706')
    axs[1].set_ylabel('Velocity (m/s)');axs[1].legend(loc='upper right')
    for name,color in zip(['FL','FR','RL','RR'],['#2563eb','#0891b2','#d97706','#9333ea']):
        axs[2].plot(t,col('normal_'+name),label=name,alpha=.8,color=color)
    axs[2].set_ylabel('Foot normal force (N)');axs[2].legend(ncol=4)
    axs[3].plot(t,col('trunk_tilt_deg'),color='#9333ea')
    axs[3].set_ylabel('Trunk tilt (deg)');axs[3].set_xlabel('Simulation time (s)')
    for ax in axs:
        ax.axvspan(tl,td,color='#e2e8f0',alpha=.5)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,4)
    fig.suptitle(f'12-joint quadruped | {summary["step_height_m"]:.2f} m step | shaded interval: flight',fontsize=15)
    fig.savefig(folder/'trajectory.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='?',default='results')
    main(p.parse_args().folder)

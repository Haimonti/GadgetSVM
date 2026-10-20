"""Render the user's three-row reference layout from inspected chart rows."""
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'deliverables/a9a_20261020/chart_data.json')
args = parser.parse_args()
rows = [r for r in json.loads(args.data.read_text(encoding='utf8')) if r['method'] != 'fedssl_tabular']
names = ['Centralized', 'SDCA', 'BDSVM (linear)', 'FedAvg', 'CoCoA', 'CoCoA+']
colors = ['#444444', '#28638E', '#B68B29', '#C37343', '#73864D', '#485B27']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'text.color':'#263139', 'axes.labelcolor':'#505B63'})
fig, axes = plt.subplots(3, 2, figsize=(16, 14.5))

def style(ax, title, xlabel):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', color='#E5E8EA', linewidth=.7)
    ax.set_axisbelow(True)

def bars(ax, kind):
    ids = list(range(len(rows))) if kind in ('accuracy', 'comm') else list(range(1,len(rows)))
    vals, labels = [], []
    for i in ids:
        r = rows[i]
        if kind == 'accuracy':
            v = (r['average_accuracy'] or 0)*100
            label = f'{v:.2f}%' if r['converged'] else 'N/A - not converged'
        elif kind == 'comm':
            v = r['total_comm_bytes']/1e6; label = f'{v:,.2f} MB'
        elif kind == 'rounds':
            v = r['rounds']; label = f'{v:,}'
        else:
            ratio = rows[0]['training_seconds'] / r['training_seconds']
            v = ratio
            label = f'{v:.3g}x' + (' [P]' if i in (3,4,5) else '')
        if kind in ('comm','rounds','ratio') and not r['converged']:
            label += ' (limit)'
        vals.append(v); labels.append(label)
    b = ax.barh(range(len(ids)), vals, height=.60, color=[colors[i] for i in ids])
    for bar, i, label in zip(b, ids, labels):
        if not rows[i]['converged']:
            bar.set_hatch('///'); bar.set_edgecolor('#666666')
        ax.annotate(label, (bar.get_width(),bar.get_y()+bar.get_height()/2),
                    xytext=(6,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_yticks(range(len(ids)), [names[i] for i in ids]); ax.invert_yaxis()
    title, xlabel = {
        'accuracy':('Final test accuracy','Accuracy (%)'),
        'comm':('Total communication at training stop','Payload sent by all workers (MB)'),
        'rounds':('Training rounds until stop','Gossip rounds'),
        'ratio':('Observed time ratio versus centralized baseline','Centralized seconds / algorithm seconds')
    }[kind]
    style(ax,title,xlabel)
    ax.set_xlim(0,112 if kind=='accuracy' else max(vals)*1.40)
    if kind=='accuracy': ax.set_xticks([0,20,40,60,80,100])

bars(axes[0,0],'accuracy'); bars(axes[0,1],'comm')
ax = axes[1,0]
ax.axhline(rows[0]['final_train_hinge'], color=colors[0], linestyle=':', label='Centralized final')
linestyles = ['-', '--', '-.', ':', '--']
for i in range(1,6):
    h = rows[i]['history']
    ax.plot([v['round'] for v in h], [v['train_hinge'] for v in h],
            color=colors[i], linestyle=linestyles[i-1], linewidth=1.8, label=names[i])
    ax.plot(rows[i]['rounds'],rows[i]['final_train_hinge'],'o',color=colors[i],markersize=3)
ax.set_xscale('log');ax.set_yscale('log')
style(ax,'Hinge loss on the training set','Gossip round (log scale)')
ax.set_ylabel('Mean training hinge loss (log scale)')
ax.grid(axis='y', color='#E5E8EA',linewidth=.7)
ax.legend(loc='upper right',ncol=2,frameon=False,fontsize=9)
bars(axes[1,1],'rounds');bars(axes[2,0],'ratio')
ax=axes[2,1];ax.axis('off')
ax.set_title('Training time and stopping status',loc='left',fontsize=14,fontweight='bold',pad=15)
data=[]
for i,r in enumerate(rows):
    status='Solver tolerance' if i==0 else 'Train plateau' if r['converged'] else 'Limit reached'
    if i in (3,4,5):status+=' [P]'
    data.append([names[i],f"{r['training_seconds']:,.3f}",status])
t=ax.table(cellText=data,colLabels=['Method','Seconds','Stop'],cellLoc='left',colLoc='left',
           colWidths=[.35,.23,.42],bbox=[0,.28,1,.70])
t.auto_set_font_size(False);t.set_fontsize(10)
for (ri,ci),cell in t.get_celld().items():
    cell.set_edgecolor('#E5E8EA');cell.set_linewidth(.5)
    if ri==0:cell.set_facecolor('#F0F2F3');cell.set_text_props(weight='bold')
ax.text(0,.17,'1x = centralized time; values below 1x are slower.\n[P] Ratios use elapsed time from overlapping runs.\nOther timings use sequential simulation on one host.\nIncludes train monitoring; this is not a multi-machine speedup.',
        transform=ax.transAxes,va='top',fontsize=9.5,color='#606C75',linespacing=1.5)
fig.suptitle('a9a · comparison preview',x=.055,y=.985,ha='left',fontsize=23,fontweight='bold')
fig.text(.055,.951,'32,561 train / 16,281 test samples · 10 workers · seed 42 · G=50 rounds · relative tolerance 0.001',fontsize=11,color='#606C75')
fig.text(.055,.025,
    'Test accuracy is evaluated only after passing the training stop rule. Hatched bars indicate the iteration limit, not convergence.\n'
    'SSL is excluded from this figure. Time ratios describe observed elapsed time, not a fair isolated speedup comparison.\n'
    'All P2P methods: adjacent 50-round windows, all peers, two stable comparisons. Communication counts directed numerical payloads, excluding headers and offline monitoring.\n'
    'Models, objectives and local compute budgets differ. FDR-SVM is not integrated and is omitted. [P] FedAvg, CoCoA and CoCoA+ overlapped in this run.',
    fontsize=9,color='#606C75',linespacing=1.6)
fig.subplots_adjust(left=.12,right=.97,top=.91,bottom=.16,wspace=.42,hspace=.46)
for ext in ['png','pdf']: fig.savefig(f'a9a_six_panel_no_ssl.{ext}',dpi=180,facecolor='white')
print('Rendered six current methods excluding SSL in the reference six-panel layout.')

"""Eight reference-layout figures for one certified common training quality."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_unified_quality import METHODS

NAMES=['Centralized','SDCA','BDSVM (linear)','FedAvg','CoCoA','CoCoA+','GADGET','CoLA','CHOCO-SVM*']
COLORS=['#444444','#28638E','#B68B29','#C37343','#73864D','#485B27','#7B559D','#248F91','#AF6B8F']


def render(root,dataset):
    root=Path(root);folder=root/'study'/dataset
    runs={m:[json.loads((folder/f'seed_{s}/{m}_timing.json').read_text()) for s in range(42,47)] for m in METHODS}
    expected=json.loads((root/'expected_data.json').read_text())[dataset]
    rows=[]
    for method,name in zip(METHODS,NAMES):
        rs=runs[method];passed=sum(r['converged'] for r in rs)
        budget=sum(r['stop_reason']=='budget_quality_unattainable' for r in rs)
        row={'method':method,'name':name,'target_passed':passed,'budget_infeasible':budget,'round_limited':5-passed-budget}
        for key in ['total_comm_bytes','rounds','training_seconds','final_train_hinge']:
            values=[r[key] for r in rs if r.get(key) is not None]
            # Do not report successful-only or trained-only five-seed means.
            row[key+'_mean']=float(np.mean(values)) if len(values)==5 else None
            row[key+'_std']=float(np.std(values,ddof=1)) if len(values)==5 else None
        acc=[100*r['average_accuracy'] for r in rs] if passed==5 else []
        row['accuracy_mean']=float(np.mean(acc)) if acc else None
        row['accuracy_std']=float(np.std(acc,ddof=1)) if acc else None
        ratios=[runs['centralized'][j]['training_seconds']/r['training_seconds'] for j,r in enumerate(rs)] if passed==5 and all(r['converged'] for r in runs['centralized']) else []
        row['ratio_mean']=float(np.mean(ratios)) if ratios else None
        row['ratio_std']=float(np.std(ratios,ddof=1)) if ratios else None
        rows.append(row)
    (folder/'quality_summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    with (folder/'quality_summary.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#263139','axes.labelcolor':'#505B63'})
    fig,axes=plt.subplots(3,2,figsize=(16,14.5))
    def style(ax,title,label):
        ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=15)
        ax.set_xlabel(label,labelpad=9);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',color='#E5E8EA',linewidth=.7);ax.set_axisbelow(True)
    def bars(ax,kind):
        ids=list(range(9)) if kind in ['accuracy','communication'] else list(range(1,9))
        key,scale={'accuracy':('accuracy',1.),'communication':('total_comm_bytes',1e-6),'rounds':('rounds',1.),'ratio':('ratio',1.)}[kind]
        values=[];errors=[];labels=[]
        for i in ids:
            r=rows[i];mean=r.get(key+'_mean');sd=r.get(key+'_std')
            values.append(0 if mean is None else mean*scale);errors.append(0 if sd is None else sd*scale)
            if mean is None:
                label='N/A: budget target infeasible' if r['budget_infeasible']==5 else f"N/A: target {r['target_passed']}/5"
            elif kind=='accuracy':label=f'{mean:.3f} +/- {sd:.3f}%'
            elif kind=='communication':label=(f'{mean*scale:,.0f} +/- {sd*scale:,.0f} MB' if mean*scale>=100000 else f'{mean*scale:,.2f} +/- {sd*scale:,.2f} MB')
            elif kind=='rounds':label=f'{mean:,.1f} +/- {sd:,.1f}'
            else:label=f'{mean:.3g}x'
            if mean is not None and r['target_passed']<5:label+=f"\ntarget {r['target_passed']}/5; cap {r['round_limited']}/5"
            labels.append(label)
        marks=ax.barh(range(len(ids)),values,xerr=errors,height=.58,color=[COLORS[i] for i in ids],error_kw={'elinewidth':1.,'capsize':2,'ecolor':'#333333'})
        for bar,i,label,err in zip(marks,ids,labels,errors):
            if rows[i]['target_passed']<5:bar.set_hatch('///');bar.set_edgecolor('#666666')
            ax.annotate(label,(bar.get_width()+err,bar.get_y()+bar.get_height()/2),xytext=(6,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_yticks(range(len(ids)),[NAMES[i] for i in ids]);ax.invert_yaxis()
        ax.set_xlim(0,119 if kind=='accuracy' else max(max(v+e for v,e in zip(values,errors))*1.65,1e-7))
        if kind=='accuracy':ax.set_xticks([0,20,40,60,80,100])
        title,label={'accuracy':('Final test accuracy at common quality','Accuracy (%)'),
          'communication':('Total communication at training stop','Payload sent by all workers (MB)'),
          'rounds':('Training rounds until target or limit','Gossip rounds'),
          'ratio':('Time ratio at common training quality','Centralized seconds / algorithm seconds')}[kind]
        style(ax,title,label)
    bars(axes[0,0],'accuracy');bars(axes[0,1],'communication')
    ax=axes[1,0];central=rows[0]['final_train_hinge_mean']
    if central is not None and central>0:ax.axhline(central,color=COLORS[0],ls=':',label='Centralized final')
    positive=[central] if central is not None and central>0 else []
    styles=['-','--','-.',':','--','-.','-',':']
    for i,method in enumerate(METHODS[1:],1):
        rs=runs[method]
        if not all(r['history'] for r in rs):
            ax.plot([],[],color=COLORS[i],ls=styles[i-1],label=NAMES[i]+' (budget limit)');continue
        grid=sorted({h['round'] for r in rs for h in r['history']})
        matrix=np.array([np.interp(grid,[h['round'] for h in r['history']],[h['train_hinge'] for h in r['history']]) for r in rs])
        mean=matrix.mean(axis=0);sd=matrix.std(axis=0,ddof=1)
        positive.extend(mean[mean>0].tolist())
        ax.plot(grid,mean,color=COLORS[i],ls=styles[i-1],lw=1.6,label=NAMES[i])
        ax.fill_between(grid,np.maximum(mean-sd,1e-12),mean+sd,color=COLORS[i],alpha=.10,lw=0)
    ax.set_xscale('log');ax.set_yscale('log')
    if positive:ax.set_ylim(min(positive)/1.7,max(positive)*4)
    style(ax,'Hinge loss on the training set','Gossip round (log scale)');ax.set_ylabel('Mean training hinge loss (log scale)')
    ax.grid(axis='y',color='#E5E8EA',linewidth=.7);ax.legend(loc='upper right',ncol=3,frameon=False,fontsize=8)
    bars(axes[1,1],'rounds');bars(axes[2,0],'ratio')
    ax=axes[2,1];ax.axis('off');ax.set_title('Training time and target status',loc='left',fontsize=14,fontweight='bold',pad=15)
    cells=[]
    for r in rows:
        value=r['training_seconds_mean'];sd=r['training_seconds_std']
        seconds='N/A' if value is None else f'{value:,.3f} +/- {sd:,.3f}'
        status=('Budget infeasible 5/5' if r['budget_infeasible']==5 else f"Target {r['target_passed']}/5; cap {r['round_limited']}/5")
        cells.append([r['name'],seconds,status])
    table=ax.table(cellText=cells,colLabels=['Method','Seconds','Status'],cellLoc='left',colLoc='left',colWidths=[.30,.35,.35],bbox=[0,.30,1,.68])
    table.auto_set_font_size(False);table.set_fontsize(8.8)
    for (ri,ci),cell in table.get_celld().items():
        cell.set_edgecolor('#E5E8EA');cell.set_linewidth(.5)
        if ri==0:cell.set_facecolor('#F0F2F3');cell.set_text_props(weight='bold')
    ax.text(0,.20,'Same three P-cores [10,11,12]; 3 threads.\n1x = centralized time; below 1x is slower.\nIsolated single-host simulation, including monitoring.\nReference preparation and final testing excluded.\nGADGET: native Python; several other loops compiled.',transform=ax.transAxes,va='top',fontsize=9.4,color='#606C75',linespacing=1.5)
    fig.suptitle(f'{dataset} · common-quality comparison',x=.055,y=.985,ha='left',fontsize=23,fontweight='bold')
    fig.text(.055,.951,f"{expected['train_samples']:,} train / {expected['test_samples']:,} test · 10 workers · random graph (k-out=3, undirected) · seeds 42-46 · mean +/- sample SD · target gap <= 0.001",fontsize=11,color='#606C75')
    fig.text(.055,.025,
      'Common objective: mean train hinge + (1e-4 / 2)||w||²; no bias. Every worker must satisfy J(w) - the same feasible dual lower bound <= 0.001.\n'
      'All methods use this quality target; native early stops and G50 stops are disabled. Common cap: 50,000 rounds. Hatched bars mark target-unmet limit costs.\n'
      'Accuracy and time-to-target ratios require 5/5 target passes; no successful-only averages. Test scores average final worker models within each seed.\n'
      'Curves: mean +/- sample SD; stopped models are held fixed. Byte counts include all directed numeric payloads and compressed indices, excluding headers/observer traffic.\n'
      'Linear BDSVM: full P=d, C=1/(2 lambda N), b=0. Local sparse/dense solvers use the same equations; native statistic gossip charges full dense matrices, including zeros.\n'
      '*CHOCO-SVM: author consensus/compression with hinge adaptation; top-20%, batch 128; gisette uses the author linear output estimator. Zero-score predictions are -1 for all methods.',
      fontsize=9,color='#606C75',linespacing=1.55)
    fig.subplots_adjust(left=.12,right=.97,top=.91,bottom=.16,wspace=.42,hspace=.46)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in [axes[0,0],axes[0,1],axes[1,1],axes[2,0]]:
        needed=ax.get_xlim()[1]
        for text in ax.texts:
            if not hasattr(text,'xy'):continue
            fraction=min(.90,(text.get_window_extent(renderer).width+12)/ax.get_window_extent(renderer).width)
            needed=max(needed,text.xy[0]/(1-fraction))
        ax.set_xlim(0,needed*1.015)
    target=root/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/(dataset+'.png'),dpi=180,facecolor='white');plt.close(fig)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    for ds in json.loads((args.root/'expected_data.json').read_text()):render(args.root,ds)

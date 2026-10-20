"""Four-panel static comparison with explicit tabular-adaptation caveats."""
import json,csv,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER=['centralized','sdca','bdsvm','fedavg','cocoa','cocoa_plus','fedssl_tabular']
NAMES=['Centralized','SDCA','BDSVM (linear)','FedAvg','CoCoA','CoCoA+','FedSSL-AMC (tabular/gossip)*']
COLORS=['#444444','#28638E','#B68B29','#C37343','#73864D','#485B27','#5995C1']


def render(directory):
    rows=[json.loads((directory/f'{m}_metrics.json').read_text(encoding='utf8')) for m in ORDER]
    hashes=[[(p['name'],p['sha256']) for p in r['source_data']] for r in rows]
    assert all(h==hashes[0] for h in hashes),'Source data mismatch'
    assert all(r['train_samples']==32561 and r['test_samples']==16281 for r in rows)
    assert rows[2]['parameters']['kernel']=='linear'
    assert all(r['adjacency']==rows[1]['adjacency'] for r in rows[1:])
    assert len({r['neighbor_schedule_sha256_first_100'] for r in rows[1:]})==1
    assert len({r['communication_policy'] for r in rows[1:]})==1
    assert rows[-1]['per_node_test_evaluations']==([1]*10 if rows[-1]['converged'] else [0]*10)
    out=directory/'preview';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#202A32','axes.labelcolor':'#39454E'})

    def bars(ax,kind):
        inds=list(range(1,7)) if kind in ('rounds','speedup') else list(range(7))
        values=[];labels=[]
        for i in inds:
            r=rows[i];c=r['converged']
            if kind=='accuracy':
                v=100*r['average_accuracy'] if c and r['average_accuracy'] is not None else 0
                label=f'{v:.2f}%' if c else 'N/A (not converged)'
            elif kind=='hinge':v=r['final_train_hinge'];label=f'{v:.4f}'
            elif kind=='communication':v=r['total_comm_bytes']/1e6;label=f'{v:,.2f} MB'
            elif kind=='rounds':v=r.get('link_cycles',r['rounds']);label=f'{v:,}'
            else:
                v=r.get('speedup_vs_centralized') or 0
                label=f'{v:.3g}x' if c and v else 'N/A (not converged)'
            if not c and kind in ['hinge','communication','rounds']:label+=' (limit)'
            values.append(v);labels.append(label)
        b=ax.barh(np.arange(len(inds)),values,color=[COLORS[i] for i in inds],height=.62)
        for bar,i,label in zip(b,inds,labels):
            if not rows[i]['converged']:bar.set_hatch('///');bar.set_edgecolor('#555555')
            ax.annotate(label,(bar.get_width(),bar.get_y()+bar.get_height()/2),xytext=(6,0),textcoords='offset points',va='center',fontsize=10)
        ax.set_yticks(np.arange(len(inds)),[NAMES[i] for i in inds]);ax.invert_yaxis()
        title,unit={'accuracy':('Final test accuracy','Accuracy (%)'),
            'hinge':('Final hinge loss on the training set','Mean hinge loss across final peer classifiers'),
            'communication':('Total communication at stop','Directed numerical payload (MB, decimal)'),
            'rounds':('PeerSim cycles until stop','Link cycles (one activation per node per cycle)'),
            'speedup':('Runtime ratio vs centralized baseline','Centralized training seconds / algorithm seconds')}[kind]
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=14);ax.set_xlabel(unit,labelpad=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#E3E6E8',linewidth=.65);ax.set_axisbelow(True)
        ax.set_xlim(0,112 if kind=='accuracy' else max(max(values)*1.4,1e-6))
        if kind=='accuracy':ax.set_xticks([0,20,40,60,80,100])
        if kind=='communication':
            ax.set_xscale('symlog',linthresh=1,linscale=1)
            ax.set_xlim(0,max(max(values)*30,10))
            ax.set_xticks([0,1,10,100,1000,10000,100000],['0','1','10','100','1,000','10,000','100,000'])
            ax.set_xlabel('MB; linear from 0 to 1, logarithmic above 1',labelpad=8)
            ax.set_title('Total communication at stop (symlog scale)',loc='left',fontsize=13,fontweight='bold',pad=14)

    fig,axes=plt.subplots(2,2,figsize=(18,11))
    for ax,kind in zip(axes.flat,['accuracy','communication','hinge','rounds']):bars(ax,kind)
    fig.suptitle('a9a · comparison with FedSSL-AMC tabular adaptation',x=.04,ha='left',fontsize=20,fontweight='bold',y=.985)
    fig.text(.04,.944,'32,561 train / 16,281 test · 10 peers · seed 42 · equal 3-thread budget · one run per method',fontsize=11,color='#505B63')
    fig.text(.04,.025,
        '* Tabular adaptation: nonlinear MLP + marginal feature corruption + frozen local linear SVMs; different capacity/objective from raw linear SVM methods.\n'
        'Accuracy bars require operational convergence; hatched bars denote the iteration limit. Training hinge uses the full train set, never the test set.\n'
        'Shared one-hop gossip: 3 neighbors/node/cycle; one SSL training round = one link cycle. Plateau: all-peer train-only monitors, G=50, tolerance .001, two stable comparisons.\n'
        'Communication includes numeric protocol metadata where counted; physical network overhead and offline monitor traffic excluded. FDR-SVM is not yet integrated.',
        fontsize=9,color='#505B63',linespacing=1.55)
    fig.subplots_adjust(left=.15,right=.97,top=.89,bottom=.20,wspace=.50,hspace=.42)
    for ext in ['png','pdf']:fig.savefig(out/f'a9a_four_panel.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));bars(ax,'speedup')
    fig.text(.05,.015,'Sequential single-host simulation; includes train monitoring and SSL final head CV. Different objectives: not an equal-quality distributed speedup.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(out/'speedup.png',dpi=180,facecolor='white');plt.close(fig)
    ssl=rows[-1];fig,ax=plt.subplots(figsize=(10,5))
    ax.plot([h['training_round'] for h in ssl['history']],[h['train_contrastive_loss'] for h in ssl['history']],color=COLORS[-1])
    ax.set(xlabel='Encoder training round',ylabel='Fixed train contrastive loss',title='a9a · FedSSL-AMC tabular adaptation (not hinge loss)')
    ax.spines[['top','right']].set_visible(False);ax.grid(color='#E3E6E8',linewidth=.65)
    fig.tight_layout();fig.savefig(out/'ssl_train_contrastive.png',dpi=180);plt.close(fig)
    fields=['method','converged','stop_reason','rounds','link_cycles','average_accuracy','accuracy_status','final_train_hinge','total_comm_bytes','training_seconds','speedup_vs_centralized']
    with (directory/'summary_with_ssl.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (directory/'chart_source.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf8')
    print('Verified seven methods, data hashes, linear BDSVM and final-only SSL test evaluations.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);render(ap.parse_args().directory)

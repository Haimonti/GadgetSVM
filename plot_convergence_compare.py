"""Reproducible training curve and final-value comparison figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'centralized': 'Centralized', 'sdca': 'SDCA', 'bdsvm': 'BDSVM',
          'fedavg': 'FedAvg', 'cocoa': 'CoCoA', 'cocoa_plus': 'CoCoA+',
          'fdr_svm': 'FDR-SVM', 'fedssl': 'FedSSL'}
COLORS = {'centralized': '#444444', 'sdca': '#28638E', 'bdsvm': '#B68B29',
          'fedavg': '#C37343', 'cocoa': '#73864D', 'cocoa_plus': '#485B27',
          'fdr_svm': '#AF6B8F', 'fedssl': '#5995C1'}
STYLES = {'sdca': '-', 'bdsvm': '--', 'fedavg': '-.', 'cocoa': ':',
          'cocoa_plus': '--', 'fdr_svm': '-', 'fedssl': '-.'}


def load_results(directory):
    return [json.loads((directory / f'{method}_metrics.json').read_text(encoding='utf-8'))
            for method in LABELS if method not in ('fdr_svm', 'fedssl')
            and (directory / f'{method}_metrics.json').exists()]


def draw_panel(ax, kind, rows):
    baseline = next((r for r in rows if r['method'] == 'centralized'), None)
    if kind == 'train_hinge':
        for row in rows:
            if row['method'] == 'centralized':
                ax.axhline(row['final_train_hinge'], color=COLORS['centralized'],
                           linestyle=':', linewidth=1.5, label='Centralized final')
                continue
            history = [h for h in row['history'] if h['train_hinge'] is not None]
            if history:
                ax.plot([h['round'] for h in history], [h['train_hinge'] for h in history],
                        label=LABELS[row['method']], color=COLORS[row['method']],
                        linestyle=STYLES[row['method']], linewidth=1.6)
        ax.set(title='Hinge loss on training set', xlabel='Training round', ylabel='Mean hinge loss')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, ncol=2, frameon=False)
    else:
        selected = [r for r in rows if kind not in ('rounds', 'speedup') or r['method'] != 'centralized']
        if kind == 'accuracy':
            values = [r['average_accuracy'] * 100 if r['average_accuracy'] is not None else 0 for r in selected]
            title, unit = 'Final test accuracy', 'Accuracy (%)'
            labels = [f'{v:.2f}%' if r['average_accuracy'] is not None else 'N/A'
                      for v, r in zip(values, selected)]
        elif kind == 'communication':
            values = [r['total_comm_bytes'] / 1e6 for r in selected]
            title, unit = 'Total communication at stopping', 'Payload sent (MB, decimal)'
            labels = [f'{v:,.2f}' for v in values]
        elif kind == 'rounds':
            values = [r['rounds'] for r in selected]
            title, unit = 'Training rounds to stopping', 'Rounds'
            labels = [f'{v:,}' for v in values]
        else:
            values = [r.get('runtime_ratio_at_stop') or 0 for r in selected]
            title, unit = 'Speed ratio vs centralized', 'Centralized time / algorithm time'
            labels = [f'{v:.4f}×' if r.get('runtime_ratio_at_stop') is not None
                      else 'N/A' for v, r in zip(values, selected)]
        positions = np.arange(len(selected))
        bars = ax.bar(positions, values, color=[COLORS[r['method']] for r in selected],
                      edgecolor='#444444', linewidth=.6, width=.65)
        for bar, row, label in zip(bars, selected, labels):
            if not row['converged']:
                bar.set_hatch('///')
                label += '\n(limit)'
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center',
                        va='bottom', fontsize=8)
        ax.set_xticks(positions, [LABELS[r['method']] for r in selected])
        ax.tick_params(axis='x', labelsize=9)
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, 100 if kind == 'accuracy' else max(max(values, default=0)*1.3, .01 if kind == 'speedup' else 1e-6))
        if kind == 'speedup':
            ax.text(.01, .97, '1× = centralized time; lower values mean slower',
                    transform=ax.transAxes, va='top', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)


def plot_results(directory):
    directory = Path(directory)
    rows = load_results(directory)
    if not rows:
        raise ValueError('No metrics found')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titleweight': 'bold', 'axes.titlesize': 12})
    plots = directory / 'plots'
    plots.mkdir(exist_ok=True)
    kinds = ['train_hinge', 'communication', 'accuracy', 'rounds', 'speedup']
    note = ('One seed; test evaluated once per final peer model after convergence; N/A if capped. Hatching = limit reached.\n'
            'Communication = directed numerical payload; metadata where counted; no physical-network overhead. Timing = isolated single-host simulation, including offline training monitoring.')
    for kind in kinds:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        draw_panel(ax, kind, rows)
        fig.suptitle(rows[0]['dataset'], fontsize=14, fontweight='bold')
        fig.text(.06, .02, note, fontsize=8, color='#555555')
        fig.tight_layout(rect=(0, .09, 1, .96))
        for ext in ('png', 'pdf'):
            fig.savefig(plots / f'{kind}.{ext}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(17, 16))
    for ax, kind in zip(axes.flat, kinds):
        draw_panel(ax, kind, rows)
    axes[2, 1].axis('off')
    parameters = rows[0]['stopping']
    axes[2, 1].text(.02, .95,
        f"Dataset: {rows[0]['dataset']}\nTrain / test: {rows[0]['train_samples']:,} / {rows[0]['test_samples']:,}\n"
        f"Seed: 42; peers: 10\nAdjacent window: {parameters['stability_window']} rounds\n"
        f"Patience G: {parameters['patience']} rounds; relative tolerance: {parameters['rtol']:g}\n\n"
        'Train hinge: mean across peer models on all train shards.\n'
        'Bars: final values at the actual stopping round.\n'
        'Hatching: did not satisfy the stopping criterion.\n'
        'A plateau is an operational rule, not an optimality certificate.\n\n'
        'FedSSL: see separately labelled tabular adaptation.\n'
        'FDR: PeerSim integration pending; excluded here.\n'
        'BDSVM uses linear budgeted features and its own regularization.\n'
        'Speed ratios compare implementations with different model objectives;\n'
        'they do not establish equal-quality or distributed acceleration.',
        transform=axes[2, 1].transAxes, va='top', fontsize=11, linespacing=1.8)
    fig.suptitle(f"{rows[0]['dataset']} · convergence comparison", fontsize=19, fontweight='bold')
    fig.text(.04, .012, note, fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .05, 1, .97), h_pad=4, w_pad=3)
    for ext in ('png', 'pdf'):
        fig.savefig(plots / f'overview.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    plot_results(Path(sys.argv[1]))

"""Render a single-dataset preview from measured convergence results only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_convergence_compare import COLORS, LABELS, STYLES, load_results


def name(method):
    return 'FedSSL-AMC*' if method == 'fedssl' else LABELS[method]


def finish(ax, title, xlabel):
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#E3E6E8', linewidth=.65)
    ax.set_axisbelow(True)


def bar_panel(ax, rows, kind):
    selected = [r for r in rows if kind not in ('rounds', 'speedup')
                or r['method'] != 'centralized']
    values, annotations = [], []
    for row in selected:
        pending = row.get('result_state') == 'pending_transfer'
        if pending:
            value, annotation = None, 'Pending result'
        elif kind == 'accuracy':
            value = row['average_accuracy']
            value = None if value is None else value * 100
            annotation = 'N/A — not converged' if value is None else f'{value:.2f}%'
        elif kind == 'communication':
            value = row['total_comm_bytes'] / 1e6
            annotation = f'{value:,.2f} MB'
        elif kind == 'rounds':
            value = row['rounds']
            annotation = f'{value:,}'
        else:
            value = row.get('runtime_ratio_at_stop')
            annotation = 'N/A' if value is None else f'{value:.3g}×'
        if not pending and not row['converged'] and kind in ('communication', 'rounds', 'speedup'):
            annotation += ' (limit)'
        values.append(0 if value is None else value)
        annotations.append(annotation)
    positions = np.arange(len(selected))
    bars = ax.barh(positions, values, height=.62,
                   color=[COLORS[r['method']] for r in selected],
                   edgecolor='white', linewidth=.6)
    for bar, row, annotation in zip(bars, selected, annotations):
        if not row['converged'] and row.get('result_state') != 'pending_transfer':
            bar.set_hatch('///')
            bar.set_edgecolor('#555555')
        ax.annotate(annotation, (bar.get_width(), bar.get_y()+bar.get_height()/2),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=10, color='#28323A')
    ax.set_yticks(positions, [name(r['method']) for r in selected])
    ax.invert_yaxis()
    titles = {
        'accuracy': ('Final test accuracy', 'Accuracy (%)'),
        'communication': ('Total communication at training stop', 'Payload sent by all workers (MB)'),
        'rounds': ('Training rounds until stop', 'Gossip rounds'),
        'speedup': ('Time ratio versus centralized baseline', 'Centralized seconds / algorithm seconds'),
    }
    finish(ax, *titles[kind])
    ax.set_xlim(0, 112 if kind == 'accuracy' else max(max(values, default=0)*1.45, 1e-6))
    if kind == 'accuracy':
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    if kind == 'speedup':
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))


def hinge_panel(ax, rows):
    for row in rows:
        method = row['method']
        if row.get('result_state') == 'pending_transfer':
            ax.plot([], [], color=COLORS[method], ls=STYLES[method],
                    label=f'{name(method)} (pending)')
            continue
        if method == 'centralized':
            ax.axhline(row['final_train_hinge'], color=COLORS[method],
                       lw=1.5, ls=':', label='Centralized final')
            continue
        history = [h for h in row['history'] if h['train_hinge'] is not None]
        if not history:
            continue
        x = np.array([h['round'] for h in history])
        y = np.array([h['train_hinge'] for h in history])
        ax.plot(x, y, color=COLORS[method], ls=STYLES[method],
                lw=1.9, label=name(method))
        ax.plot(x[-1], y[-1], 'o', color=COLORS[method], ms=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    finish(ax, 'Hinge loss on the training set', 'Gossip round (log scale)')
    ax.set_ylabel('Mean training hinge loss (log scale)')
    ax.grid(axis='y', color='#E3E6E8', linewidth=.65)
    ax.legend(fontsize=9, ncol=2, frameon=False, loc='upper right')


def timing_panel(ax, rows):
    ax.axis('off')
    ax.set_title('Training time and stopping status', loc='left',
                 fontsize=13, fontweight='bold', pad=14)
    cells = []
    for row in rows:
        seconds = row.get('isolated_training_seconds')
        cells.append([name(row['method']), 'N/A' if seconds is None else f'{seconds:,.3f}',
                      'Pending transfer' if row.get('result_state') == 'pending_transfer'
                      else 'Solver tolerance' if row['method'] == 'centralized' and row['converged']
                      else 'Train plateau' if row['converged'] else 'Limit reached'])
    table = ax.table(cellText=cells, colLabels=['Method', 'Seconds', 'Stop'],
                     colWidths=[.34, .23, .43], cellLoc='left', colLoc='left',
                     bbox=[0, .28, 1, .70])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#E3E6E8')
        cell.set_linewidth(.5)
        cell.set_facecolor('#F2F4F5' if i == 0 else 'white')
        if i == 0:
            cell.set_text_props(fontweight='bold')
    ax.text(0, .18,
            '1× = centralized time; values below 1× are slower.\n'
            'Timing is sequential simulation on one host, with an equal\n'
            'thread budget. It includes training monitoring.\n'
            'Models/objectives differ; this is not a multi-machine speedup.',
            transform=ax.transAxes, fontsize=9, va='top', linespacing=1.5,
            color='#505B63')


def render(directory):
    rows = load_results(directory)
    if len(rows) != 6:
        raise ValueError(f'Expected 6 validated comparison entries including centralized, found {len(rows)}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': '#202A32', 'axes.labelcolor': '#39454E'})
    plots = directory / 'preview'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14.5))
    bar_panel(axes[0, 0], rows, 'accuracy')
    bar_panel(axes[0, 1], rows, 'communication')
    hinge_panel(axes[1, 0], rows)
    bar_panel(axes[1, 1], rows, 'rounds')
    bar_panel(axes[2, 0], rows, 'speedup')
    timing_panel(axes[2, 1], rows)
    fig.suptitle(f"{rows[0]['dataset']} · comparison preview", x=.055, ha='left',
                 fontsize=21, fontweight='bold', y=.985)
    rule = rows[0]['stopping']
    fig.text(.055, .952,
             f"{rows[0]['train_samples']:,} train / {rows[0]['test_samples']:,} test samples · "
             f"10 workers · seed 42 · G={rule['patience']} rounds · "
             f"relative tolerance {rule['rtol']:g}", fontsize=11, color='#505B63')
    pending_note = ('FedAvg result transfer is pending: the private compute host is offline. No FedAvg values are imputed.\n'
                    if any(r.get('result_state') == 'pending_transfer' for r in rows) else '')
    fig.text(.055, .022,
             pending_note +
             'Test accuracy is evaluated once per final worker model after the training stop. '
             'Training loss is averaged across worker models on all train shards.\n'
             'Plateau is an operational stopping rule. Hatched bars indicate a limit, not convergence. '
             'Legacy FDR and SVD-based FedSSL entries are excluded pending paper alignment.\n'
             'FedAvg: adjacent 50-round objective means, two stable comparisons. '
             'Other P2P methods: 10-round smoothing and 50-round objective patience.\n'
             'Communication counts directed numerical payloads, excluding headers and offline monitoring traffic. '
             'BDSVM uses its native kernel. SSL is validated separately on I/Q data.',
             fontsize=9, color='#505B63', linespacing=1.6)
    fig.subplots_adjust(left=.12, right=.97, top=.91, bottom=.16,
                        wspace=.42, hspace=.42)
    fig.savefig(plots / f'{rows[0]["dataset"]}_comparison_preview.png', dpi=180, facecolor='white')
    plt.close(fig)
    for kind in ('accuracy', 'communication', 'rounds', 'speedup', 'train_hinge'):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        if kind == 'train_hinge':
            hinge_panel(ax, rows)
        else:
            bar_panel(ax, rows, kind)
        fig.suptitle(rows[0]['dataset'], fontsize=16, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(plots / f'{kind}.png', dpi=180, facecolor='white')
        plt.close(fig)
    print(f'Rendered {plots}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)

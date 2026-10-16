"""Validate and package the completed full-data PeerSim runs by algorithm."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASETS = ['covtype', 'gisette', 'real-sim', 'rcv1', 'ijcnn1', 'a9a', 'w8a', 'webspam']
LABELS = {'fedssl_amc': 'FedSSL-AMC', 'fdr_svm': 'FDR-SVM'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='2026-10-15')
    args = ap.parse_args()
    rows, files = [], {m: [] for m in LABELS}
    manifest = json.loads((ROOT/'linear_data_manifest.json').read_text())
    for name, rec in manifest.items():
        h = hashlib.sha256()
        with (ROOT/'data'/'processed'/name).open('rb') as fh:
            for block in iter(lambda: fh.read(8*1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == rec['sha256'], f'Dataset hash differs: {name}'
    for ds in DATASETS:
        out = ROOT/'results'/f'additional_{ds}_{args.tag}'
        for method in LABELS:
            path = out/f'{method}_metrics.json'
            d = json.loads(path.read_text())
            assert d['method'] == method and d['config']['DATASET'] == ds
            assert d['engine'] == 'src.peersim_python.cdsim.CDSimulator'
            assert d['cycles_requested'] == 5000 and 1 <= d['stopped_at'] <= 5000
            cfg = d['config']
            assert (cfg['SEED'], cfg['NUM_WORKERS'], cfg['GOSSIP_K']) == (42, 10, 3)
            assert cfg['TOPOLOGY'] == 'random_kout' and cfg['EVAL_EVERY'] == 10
            assert cfg['SDCA_LOCAL_STEPS'] == 4096 and cfg['LAMBDA'] == 1e-4
            histories = d['per_node_metrics']
            assert len(histories) == 10 and all(len(h) >= 2 for h in histories)
            for i, history in enumerate(histories):
                assert history[-1]['round'] == d['stopped_at']
                assert np.all(np.diff([r['round'] for r in history]) > 0)
                assert np.all(np.diff([r['comm_bytes'] for r in history]) >= 0)
                assert np.isfinite(history[-1]['accuracy'])
                assert np.isfinite(history[-1]['hinge_loss'])
                assert np.isfinite(history[-1]['consensus_error'])
                assert np.isnan(history[-1]['duality_gap'])
                assert history[-1]['comm_bytes'] == d['per_node_comm_bytes'][i]
                assert np.isclose(history[-1]['accuracy'], d['per_node_accuracy'][i])
            assert sum(d['per_node_comm_bytes']) == d['total_comm_bytes']
            assert np.isclose(np.mean(d['per_node_accuracy']), d['average_accuracy'])
            converged = all(c is not None for c in d['per_node_stop_cycle'])
            if not converged:
                assert d['stopped_at'] == 5000
            rows.append(dict(dataset=ds, method=method, accuracy=d['average_accuracy'],
                             cycles=d['stopped_at'], converged=converged,
                             communication_mb=d['total_comm_bytes']/1e6,
                             elapsed_seconds=d['elapsed_seconds']))
            paths = [path, out/'plots'/f'merged_with_{method}.png']
            paths += [out/'plots'/f'merged_{metric}_{method}.png' for metric in
                      ['accuracy', 'hinge_loss', 'consensus_error', 'comm_bytes']]
            paths += [out/f'{m}_metrics.json' for m in ['sdca', 'bdsvm_linear']]
            for p in paths:
                assert p.exists(), p
                if p.suffix == '.png':
                    with Image.open(p) as im:
                        im.verify()
                files[method].append((p, f'{ds}/{p.relative_to(out).as_posix()}'))
    overview = ROOT/'results'/f'plots_additional_{args.tag}'
    for method in LABELS:
        for metric in ['accuracy', 'hinge_loss', 'comm_bytes']:
            p = overview/f'overview_{metric}_{method}.png'
            with Image.open(p) as im:
                im.verify()
            files[method].append((p, f'overview/{p.name}'))
        rows_m = [r for r in rows if r['method'] == method]
        s = io.StringIO(newline='')
        w = csv.DictWriter(s, fieldnames=list(rows_m[0]))
        w.writeheader()
        w.writerows(rows_m)
        name = f'{LABELS[method]}_PeerSim_all8_{args.tag}.zip'
        destination = ROOT.parent/name
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('summary.csv', s.getvalue())
            z.write(ROOT/'ADDITIONAL_EXPERIMENTS.md', 'EXPERIMENT_DETAILS.md')
            z.write(ROOT/'linear_data_manifest.json', 'data_manifest.json')
            for p, archive_path in files[method]:
                z.write(p, archive_path)
            note = f'# {LABELS[method]} — PeerSim 全部 8 个数据集\n\n'
            note += '每个数据集的 plots 文件夹有四指标合图及四张单指标图；overview 是八数据集总览。图中附有已有 SDCA 与线性 BDSVM 基线。\n\n'
            note += '数据文件 SHA-256 与此前全量实验清单全部一致。10 个节点，seed=42，random_kout(k=3)，最多 5000 轮。\n\n'
            note += '| 数据集 | 测试准确率 | 执行轮数 | 停止状态 |\n|---|---:|---:|---|\n'
            for r in rows_m:
                status = '满足迭代稳定条件' if r['converged'] else '达到 5000 轮上限，未满足稳定条件'
                note += f'| {r["dataset"]} | {r["accuracy"]:.2%} | {r["cycles"]} | {status} |\n'
            note += '\n本次运行的是仓库已有算法的 PeerSim 适配版本；具体近似、FedSSL 坐标修复和停止规则见 EXPERIMENT_DETAILS.md。\n'
            z.writestr('结果说明.md', note)
        print(destination)
    validation = dict(runs=len(rows), datasets=len(DATASETS), methods=list(LABELS),
                      verified_dataset_files=len(manifest),
                      packaged_plots_per_method=43, status='passed', results=rows)
    (ROOT/'results'/f'additional_validation_{args.tag}.json').write_text(
        json.dumps(validation, indent=2), encoding='utf-8')
    print('Validated all 16 full-data runs and 86 packaged PNG figures')


if __name__ == '__main__':
    main()

"""Exactly eight four-metric figures per algorithm, with total bytes below."""
import argparse
import json
from pathlib import Path
import zipfile

from PIL import Image
from plot_merged import METHODS, draw

ROOT = Path(__file__).resolve().parent
DATASETS = ['covtype', 'gisette', 'real-sim', 'rcv1', 'ijcnn1', 'a9a', 'w8a', 'webspam']
LABELS = {'fedssl_amc': 'FedSSL-AMC', 'fdr_svm': 'FDR-SVM'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='2026-10-15')
    args = ap.parse_args()
    for method, label in LABELS.items():
        out = ROOT/'results'/f'eight_figures_{args.tag}'/method
        out.mkdir(parents=True, exist_ok=True)
        keys = ['sdca', 'bdsvm_linear', method]
        methods = [m for m in METHODS if m[0] in keys]
        paths = []
        for ds in DATASETS:
            source = ROOT/'results'/f'additional_{ds}_{args.tag}'
            data = {m: json.loads((source/f'{m}_metrics.json').read_text()) for m in keys}
            path = draw(methods, data, out, f'{method}_{ds}', '',
                        single_panels=False, footer_method=method)
            with Image.open(path) as im:
                im.verify()
            paths.append(path)
        archive = ROOT.parent/f'{label}_8plots_4metrics_{args.tag}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in paths:
                z.write(path, f'{path.stem.removeprefix("merged_")}.png')
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == 8 and z.testzip() is None
        print(f'{label}: exactly 8 figures -> {archive}')


if __name__ == '__main__':
    main()

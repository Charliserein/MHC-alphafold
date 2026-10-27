#!/usr/bin/env python3
"""离线复制已验证的 Python 环境，形成独立副本；不复制 Torch/Triton。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def clone(source, target):
    source = source.resolve()
    root = target.resolve()
    source_base = Path(subprocess.check_output([str(source / 'bin/python'), '-c',
                                              'import sys; print(sys.base_prefix)'], text=True).strip())
    runtime = root / '.runtime/python'
    shutil.copytree(source_base, runtime, dirs_exist_ok=True, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    venv = root / '.venv'
    subprocess.run([str(runtime / 'bin/python3.10'), '-m', 'venv', '--copies', str(venv)], check=True)
    source_site = source / 'lib/python3.10/site-packages'
    dest_site = venv / 'lib/python3.10/site-packages'
    excluded = ('torch', 'torchgen', 'functorch', 'triton', '__editable__', 'csb_pandora')
    copied = []
    for path in sorted(source_site.iterdir()):
        if path.name.startswith(excluded) or path.name in ('__pycache__', '_virtualenv.pth', '_virtualenv.py'):
            continue
        if path.is_dir():
            shutil.copytree(path, dest_site / path.name, dirs_exist_ok=True, symlinks=False,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(path, dest_site / path.name)
        copied.append(path.name)
    (root / 'validation').mkdir(exist_ok=True)
    (root / 'validation/runtime-copy.json').write_text(json.dumps({
        'method': 'independent_regular_file_copy', 'python_version': sys.version,
        'excluded': list(excluded), 'copied_site_entries': copied}, indent=2) + '\n')
    print(f'独立环境已复制到 {venv}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-venv', type=Path, required=True)
    p.add_argument('--target', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    clone(args.source_venv, args.target)

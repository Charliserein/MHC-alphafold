#!/usr/bin/env python3
"""核对 AFfine 预测代码未在抽取过程中改变，资源 SHA256 与来源一致。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(source):
    mismatches = []
    checked = 0
    for path in (ROOT / 'AFfine').rglob('*'):
        if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in path.parts:
            relative = path.relative_to(ROOT)
            if digest(path) != digest(source / relative):
                mismatches.append(str(relative))
            checked += 1
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for item in manifest['resources']:
        if digest(ROOT / item['path']) != item['sha256']:
            mismatches.append(item['path'])
    external_links = []
    for folder in ('AFfine', 'PANDORA', 'resources', '.venv', '.runtime'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
                external_links.append(str(path.relative_to(ROOT)))
    report = dict(passed=not mismatches and not external_links, af2_files_compared=checked,
                  resource_files_checked=len(manifest['resources']), mismatches=mismatches,
                  external_symlinks=external_links)
    (ROOT / 'validation/source-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    raise SystemExit(verify(p.parse_args().source.resolve()))

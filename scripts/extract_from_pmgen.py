#!/usr/bin/env python3
"""从已有 PMGen 安装提取预测实现；不修改来源目录。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def copy_tree(source, target, **kwargs):
    shutil.copytree(source, target, dirs_exist_ok=True, ignore_dangling_symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.egg-info'), **kwargs)


def extract(source, target):
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ('AFfine', 'PANDORA'):
        copy_tree(source / name, target / name)
        copied.append(name)
    for filename in ('run_finetuning.py',):
        (target / 'AFfine' / filename).unlink(missing_ok=True)
    (target / 'licenses').mkdir(exist_ok=True)
    shutil.copy2(source / 'LICENSE', target / 'licenses/PMGen-LICENSE')
    (target / 'data').mkdir(exist_ok=True)
    shutil.copy2(source / 'data/all_templates.csv', target / 'data/all_templates.csv')
    for name in ('tools', 'pandora_db'):
        copy_tree(source / 'deployment' / name, target / 'resources' / name)
    config = target / 'PANDORA/PANDORA/__init__.py'
    code = config.read_text()
    code = code.replace("PANDORA_data = os.path.expanduser(data_folder)",
        "# MHCalphafold: 数据库随独立目录迁移；环境变量可覆盖。\n"
        "PANDORA_data = os.path.abspath(os.path.expanduser(os.environ.get(\n"
        "    'MHC_AF_DATABASE', str(Path(__file__).resolve().parents[2] / 'resources' / 'pandora_db'))))")
    config.write_text(code)
    (target / 'PANDORA/PANDORA/config.json').write_text('{"data_folder_name": "resources/pandora_db"}\n')
    pmhc = target / 'PANDORA/PANDORA/PMHC/PMHC.py'
    pmhc.write_text(pmhc.read_text().replace("re.sub('[^a-zA-Z0-9 \\n\\.]', '_', id)",
                                           "re.sub(r'[^a-zA-Z0-9_.-]', '_', id)"))

    # 保留两种结构建模类；不带入肽设计、突变筛选、BioEmu 等入口。
    text = (source / 'run_utils.py').read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name in ('run_PMGen_modeling', 'run_PMGen_wrapper')]
    imports = '''"""从 PMGen 提取的结构建模实现；上游来源见 SOURCE_MANIFEST.json。"""
import ast
import concurrent.futures
import contextlib
import glob
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import psutil
from PANDORA import Target, Pandora, Database
from utils import processing_functions
warnings.filterwarnings('ignore', category=FutureWarning)

'''
    backend = imports + '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in selected)
    backend = backend.replace('class run_PMGen_modeling()', 'class StructureModeler()')
    backend = backend.replace('class run_PMGen_wrapper()', 'class StructureBatch()')
    backend = backend.replace('run_PMGen_modeling(', 'StructureModeler(')
    backend = backend.replace('template_csv_path="data/all_templates.csv"',
        'template_csv_path=str(Path(__file__).resolve().parents[1] / "data/all_templates.csv")')
    # 上游不论选择何种模型都要求存在微调文件；独立版只对 _ft 分支检查。
    backend = backend.replace('assert os.path.isfile(\n            self.fine_tuned_model_path)',
        "assert not any('_ft' in name for name in self.models) or os.path.isfile(\n            self.fine_tuned_model_path)")
    (target / 'mhcalphafold').mkdir(exist_ok=True)
    (target / 'mhcalphafold/backend.py').write_text(backend)

    # 根据所抽取类的引用，计算工具函数的传递依赖闭包。
    text = (source / 'utils/processing_functions.py').read_text()
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    used = {n.attr for n in ast.walk(ast.parse(backend)) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == 'processing_functions'}
    used.update(('anchor_combinations_mhc1', 'anchor_combinations_mhc2'))
    while True:
        more = {n.id for name in used for n in ast.walk(functions[name])
                if isinstance(n, ast.Name) and n.id in functions}
        if more <= used:
            break
        used |= more
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    source_lines = text.splitlines(keepends=True)
    nodes = sorted(imports + [functions[name] for name in used], key=lambda n: n.lineno)
    extracted = '"""从 PMGen 提取的结构预测工具函数；不含肽生成流程。"""\n'
    extracted += '\n\n'.join(''.join(source_lines[n.lineno-1:n.end_lineno]) for n in nodes)
    extracted = extracted.replace('template_csv_path="../data/templates/all_templates.csv"',
        'template_csv_path=os.path.join(pmgen_abs_dir, "data/all_templates.csv")')
    (target / 'utils').mkdir(exist_ok=True)
    (target / 'utils/__init__.py').write_text('')
    (target / 'utils/processing_functions.py').write_text(extracted)

    resources = []
    for folder in (target / 'AFfine/af_params', target / 'resources'):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                resources.append({'path': str(path.relative_to(target)), 'bytes': path.stat().st_size,
                                  'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                                  if hasattr(hashlib, 'file_digest') else sha256(path)})
    manifest = {'name': 'MHCalphafold', 'source_repository': 'https://github.com/soedinglab/PMGen',
                'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                'extracted_classes': [n.name for n in selected], 'processing_functions': sorted(used),
                'source_local_compatibility_patch': subprocess.check_output(['git', '-C', str(source), 'diff', '--', 'AFfine', 'run_utils.py'], text=True),
                'resources': resources}
    (target / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    extract(args.source.resolve(), args.target.resolve())

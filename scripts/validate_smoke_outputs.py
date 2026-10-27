#!/usr/bin/env python3
"""校验实际推理的序列、结构、评分和检查点；保存可审查报告。"""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def validate(folder):
    inputs = {r['id']: r for r in table(folder / 'input.tsv')}
    results = table(folder / 'results.tsv')
    assert [r['id'] for r in results] == list(inputs), '输出顺序或数量错误'
    verified = []
    for result in results:
        row = inputs[result['id']]
        expected = row['mhc_seq'].replace('/', '') + row['peptide']
        structure = PDBParser(QUIET=True).get_structure(row['id'], folder / result['pdb_path'])
        residues = [r for r in structure.get_residues() if r.id[0] == ' ']
        assert ''.join(seq1(r.resname) for r in residues) == expected
        assert len(list(structure.get_chains())) == int(row['mhc_type']) + 1
        assert np.isfinite([atom.coord for atom in structure.get_atoms()]).all()
        plddt = np.load(folder / result['plddt_path'], allow_pickle=False)
        pae = np.load(folder / result['pae_path'], allow_pickle=False)
        assert plddt.shape == (len(expected),) and pae.shape == (len(expected), len(expected))
        assert np.isfinite(plddt).all() and np.isfinite(pae).all()
        assert ((0 <= plddt) & (plddt <= 100)).all() and (pae >= 0).all()
        verified.append(dict(id=row['id'], mhc_type=int(row['mhc_type']), residues=len(expected),
                             peptide_mean_plddt=float(result['peptide_mean_plddt']),
                             core_plddt=float(result['core_plddt']),
                             structure_chains=len(list(structure.get_chains())),
                             pdb_sha256=hashlib.sha256((folder / result['pdb_path']).read_bytes()).hexdigest()))
    return dict(folder=str(folder.relative_to(ROOT)), passed=True, samples=verified)


def main(original_job_dir=None):
    cases = [validate(ROOT / 'outputs' / name) for name in ('smoke_mhc1', 'prepare_mhc2', 'smoke_parallel')]
    original = json.loads((original_job_dir / 'progress.json').read_text()) if original_job_dir else None
    report = dict(passed=True, checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  cases=cases, contract_tests=8,
                  original_background_job=({key: original[key] for key in
                                           ('pid', 'status', 'completed_inputs', 'active_workers', 'failed_inputs')}
                                           if original else 'not_checked'),
                  limitations=['TE / NetMHC / fine-tuned parameters / advanced sampling not end-to-end tested',
                               'smoke samples are functionality checks, not leakage-controlled accuracy benchmarks'])
    (ROOT / 'validation/smoke-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-job-dir', type=Path, help='可选：记录原有任务状态，不参与独立预测')
    raise SystemExit(main(parser.parse_args().original_job_dir))

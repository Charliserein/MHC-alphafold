"""预测产物校验、置信度计算与原子保存。"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path


RESULT_FIELDS = ['id', 'peptide', 'mhc_type', 'selected_candidate', 'model', 'anchors',
                 'core_plddt', 'peptide_mean_plddt', 'overall_mean_plddt', 'ptm',
                 'pdb_path', 'plddt_path', 'pae_path', 'ptm_path']


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def write_table(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def trim_scores(plddt, pae, length):
    import numpy as np
    if plddt.ndim != 1 or len(plddt) < length or pae.shape != (len(plddt), len(plddt)):
        raise ValueError('AF2 分数数组长度不足或 PAE 维度不匹配')
    plddt, pae = plddt[:length], pae[:length, :length]
    if not np.isfinite(plddt).all() or not np.isfinite(pae).all():
        raise ValueError('真实残基的预测分数包含非有限值')
    if (plddt < 0).any() or (plddt > 100).any() or (pae < 0).any():
        raise ValueError('预测分数超出有效范围')
    return plddt, pae


def validate_candidate(row, pdb):
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    prefix = str(pdb.with_suffix(''))
    plddt = np.load(prefix + '_plddt.npy', allow_pickle=False)
    pae = np.load(prefix + '_predicted_aligned_error.npy', allow_pickle=False)
    ptm = np.load(prefix + '_ptm.npy', allow_pickle=False)
    expected = row['mhc_seq'].replace('/', '') + row['peptide']
    plddt, pae = trim_scores(plddt, pae, len(expected))
    if ptm.size != 1 or not np.isfinite(ptm).all() or (ptm < 0).any() or (ptm > 1).any():
        raise ValueError('pTM 分数无效')
    structure = PDBParser(QUIET=True).get_structure(row['id'], pdb)
    residues = [r for r in structure.get_residues() if r.id[0] == ' ']
    if ''.join(seq1(r.resname) for r in residues) != expected:
        raise ValueError(f'{pdb}: PDB 残基序列与输入不一致')
    coordinates = np.array([a.coord for a in structure.get_atoms()])
    if not coordinates.size or not np.isfinite(coordinates).all():
        raise ValueError(f'{pdb}: 原子坐标无效')
    anchors = [int(a) for a in row['anchors'].split(';')]
    mhc_len = len(row['mhc_seq'].replace('/', ''))
    start = mhc_len + max(0, anchors[0] - 2)
    end = mhc_len + anchors[-1] - 1
    if start >= end:
        raise ValueError('核心评分区域为空')
    model = 'model_' + pdb.stem.rsplit('_model_', 1)[-1]
    return dict(row, pdb=pdb, model=model, plddt=plddt, pae=pae, ptm_array=ptm,
                core_plddt=float(plddt[start:end].mean()),
                peptide_mean_plddt=float(plddt[-len(row['peptide']):].mean()),
                overall_mean_plddt=float(plddt.mean()), ptm=float(ptm.reshape(-1)[0]))


def normalized_pdb(row, pdb):
    """按输入顺序导出独立链；只改 PDB 标识和编号，不改坐标。"""
    from Bio.PDB import PDBIO, PDBParser
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
    from Bio.SeqUtils import seq1
    source = PDBParser(QUIET=True).get_structure('prediction', pdb)
    residues = [r for r in source.get_residues() if r.id[0] == ' ']
    chains = row['mhc_seq'].split('/') + [row['peptide']]
    if ''.join(seq1(r.resname) for r in residues) != ''.join(chains):
        raise ValueError('分链时 PDB 序列与输入不符')
    structure = Structure(row['id'])
    model = Model(0)
    structure.add(model)
    offset = 0
    for index, sequence in enumerate(chains):
        chain = Chain(chr(ord('A') + index))
        model.add(chain)
        for position, residue in enumerate(residues[offset:offset + len(sequence)], 1):
            copied = residue.copy()
            copied.id = (' ', position, ' ')
            chain.add(copied)
        offset += len(sequence)
    writer = PDBIO()
    writer.set_structure(structure)
    class PDBBuffer(io.StringIO):
        # Biopython 1.80 会在 save() 内对调用方传入的 handle 使用 with。
        def __exit__(self, exc_type, exc_value, traceback):
            return False

    stream = PDBBuffer()
    writer.save(stream)
    content = stream.getvalue().encode()
    stream.close()
    return content


def save_best(root, row, candidates, fingerprint):
    import numpy as np
    winner = max(candidates, key=lambda r: r['core_plddt'])
    result = {name: winner[name] for name in ('peptide', 'mhc_type', 'model', 'anchors',
              'core_plddt', 'peptide_mean_plddt', 'overall_mean_plddt', 'ptm')}
    result.update(id=row['id'], selected_candidate=winner['id'])
    result['pdb_path'] = f'best_structures/{row["id"]}.pdb'
    atomic_bytes(root / result['pdb_path'], normalized_pdb(row, winner['pdb']))
    for key, values in (('plddt', winner['plddt']), ('pae', winner['pae']), ('ptm', winner['ptm_array'])):
        result[key + '_path'] = f'scores/{row["id"]}_{key}.npy'
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        atomic_bytes(root / result[key + '_path'], stream.getvalue())
    result['run_fingerprint'] = fingerprint
    result['file_sha256'] = {result[k]: hashlib.sha256((root / result[k]).read_bytes()).hexdigest()
                             for k in ('pdb_path', 'plddt_path', 'pae_path', 'ptm_path')}
    write_json(root / 'checkpoints' / f'{row["id"]}.json', result)
    return result


def load_checkpoint(root, identifier, fingerprint):
    path = root / 'checkpoints' / f'{identifier}.json'
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        if result['run_fingerprint'] != fingerprint or result['id'] != identifier:
            return None
        if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
               for p, digest in result['file_sha256'].items()):
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None

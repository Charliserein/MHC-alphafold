"""输入标准化和锚点候选展开；不加载 AF2 或 GPU。"""
import csv
from pathlib import Path
import re

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
FIELDS = ['id', 'peptide', 'mhc_seq', 'mhc_type', 'anchors']


def sequence(value, name):
    value = ''.join(str(value).split()).upper()
    if not value or set(value) - AMINO_ACIDS:
        raise ValueError(f'{name} 必须是仅含 20 种标准氨基酸字母的非空序列')
    return value


def read_fasta(path):
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            records.append('')
        elif not records:
            raise ValueError('MHC FASTA 必须包含 > 标题')
        else:
            records[-1] += line
    if not records:
        raise ValueError('MHC FASTA 为空')
    return '/'.join(records)


def normalize(row):
    identifier = str(row.get('id', '')).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', identifier):
        raise ValueError('id 必须以字母或数字开头，长度至多 100，仅含字母、数字、_、.、-')
    try:
        kind = int(row['mhc_type'])
    except (KeyError, ValueError, TypeError):
        raise ValueError(f'{identifier}: mhc_type 必须为 1 或 2') from None
    if kind not in (1, 2):
        raise ValueError(f'{identifier}: mhc_type 必须为 1 或 2')
    peptide = sequence(row.get('peptide', ''), f'{identifier}.peptide')
    chains = [sequence(s, f'{identifier}.mhc_seq') for s in str(row.get('mhc_seq', '')).split('/')]
    if len(chains) != kind:
        raise ValueError(f'{identifier}: MHC-I 需要一条 MHC 链；MHC-II 需要 Alpha/Beta 两条链')
    if any(len(chain) < 60 for chain in chains):
        raise ValueError(f'{identifier}: 请提供 MHC 结合域的真实序列，不能用 34 aa 伪序列代替')
    anchor_text = str(row.get('anchors') or '').strip()
    if anchor_text:
        try:
            anchors = [int(s.strip()) for s in anchor_text.split(';')]
        except ValueError:
            raise ValueError(f'{identifier}: anchors 应为分号分隔的整数') from None
        if len(anchors) != (2 if kind == 1 else 4) or anchors != sorted(set(anchors)):
            raise ValueError(f'{identifier}: MHC-I 需要 2 个、MHC-II 需要 4 个严格递增的锚点')
        if anchors[0] < 1 or anchors[-1] > len(peptide):
            raise ValueError(f'{identifier}: 锚点必须在 1..{len(peptide)} 范围内')
        anchor_text = ';'.join(map(str, anchors))
    return dict(id=identifier, peptide=peptide, mhc_seq='/'.join(chains), mhc_type=kind, anchors=anchor_text)


def read_inputs(args):
    if args.input:
        if any((args.peptide, args.mhc_seq, args.mhc_fasta, args.anchors)):
            raise ValueError('--input 不能与单条样本的序列参数同时使用')
        path = Path(args.input).resolve()
        with path.open(newline='', encoding='utf-8-sig') as stream:
            header = stream.readline()
            stream.seek(0)
            rows = list(csv.DictReader(stream, delimiter='\t' if '\t' in header else ','))
    else:
        if not args.peptide or not (args.mhc_seq or args.mhc_fasta) or not args.mhc_type:
            raise ValueError('请提供 --input，或同时提供 --peptide、--mhc-type 和 --mhc-seq/--mhc-fasta')
        if args.mhc_seq and args.mhc_fasta:
            raise ValueError('--mhc-seq 与 --mhc-fasta 只能选一个')
        rows = [dict(id=args.id, peptide=args.peptide, mhc_type=args.mhc_type,
                     mhc_seq=args.mhc_seq or read_fasta(args.mhc_fasta), anchors=args.anchors)]
    if not rows:
        raise ValueError('输入没有样本')
    normalized = [normalize(row) for row in rows]
    ids = [row['id'] for row in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError('输入存在重复 id，请先为每条样本分配唯一 id')
    return normalized


def anchor_combinations(length, kind):
    if kind == 1:
        return [(a, b) for a in range(1, length + 1)
                for b in range(a + 6, length + 1)]
    return [(a, a + 3, a + 5, a + 8) for a in range(1, length - 7)]


def expand(rows, mode='auto'):
    expanded = []
    for row in rows:
        if mode == 'enumerate' or (mode == 'auto' and not row['anchors']):
            candidates = [';'.join(map(str, a)) for a in anchor_combinations(len(row['peptide']), row['mhc_type'])]
            if not candidates:
                raise ValueError(f'{row["id"]}: 肽长度不足，无法枚举锚点')
        elif row['anchors']:
            candidates = [row['anchors']]
        elif mode == 'netmhc':
            candidates = ['']
        else:
            raise ValueError(f'{row["id"]}: --anchor-mode given 要求给定 anchors')
        for index, anchors in enumerate(candidates):
            expanded.append(dict(row, source_id=row['id'], id=f'{row["id"]}__a{index:04d}', anchors=anchors))
    return expanded

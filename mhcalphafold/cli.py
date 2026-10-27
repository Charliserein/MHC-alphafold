"""独立的结构预测命令行入口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from .inputs import FIELDS, expand, read_inputs
from .outputs import RESULT_FIELDS, load_checkpoint, save_best, validate_candidate, write_json, write_table

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def parser():
    p = argparse.ArgumentParser(description='MHCalphafold：独立的肽–MHC AlphaFold2 结构预测')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='检查 Python、模板库、权重和 GPU')
    raw = sub.add_parser('af2', help='将其余参数直接传入完整 AFfine 预测入口')
    raw.add_argument('af2_args', nargs=argparse.REMAINDER)
    for command, description in [('prepare', '只准备模板和 AF2 输入，不运行 GPU'),
                                 ('predict', '准备模板、预测结构并按核心 pLDDT 选优')]:
        s = sub.add_parser(command, help=description)
        s.add_argument('--input', help='TSV/CSV：id,peptide,mhc_seq,mhc_type,anchors')
        s.add_argument('--peptide')
        s.add_argument('--mhc-seq', help='MHC-I：一条结合域；MHC-II：Alpha/Beta')
        s.add_argument('--mhc-fasta', help='MHC-I：一条 FASTA；MHC-II：先 Alpha 后 Beta')
        s.add_argument('--mhc-type', type=int, choices=[1, 2])
        s.add_argument('--id', default='sample')
        s.add_argument('--anchors', help='肽内 1-based 锚点，例如 2;9')
        s.add_argument('--anchor-mode', choices=['auto', 'given', 'enumerate', 'netmhc'], default='auto',
                       help='auto：已有锚点直接使用，否则枚举；netmhc 需要授权工具')
        s.add_argument('--output', required=True)
        s.add_argument('--mode', choices=['ig', 'te'], default='ig')
        s.add_argument('--models', nargs='+', default=['model_2_ptm'])
        s.add_argument('--params-dir', default=str(ROOT / 'AFfine/af_params/params_original'))
        s.add_argument('--fine-tuned-params', default=str(ROOT / 'AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl'))
        s.add_argument('--num-templates', type=int, default=4)
        s.add_argument('--num-recycles', type=int, default=3)
        s.add_argument('--batch-size', type=int, default=50, help='每次 AF2 进程处理的原始样本数')
        s.add_argument('--resume', action='store_true', help='校验检查点后跳过已完成样本')
        s.add_argument('--return-all-outputs', action='store_true')
        s.add_argument('--sampling-mode', action='store_true', help='上游高级结构采样接口，非默认路线')
        s.add_argument('--n-times-sampling', type=int, default=50)
        s.add_argument('--sampling-seed', type=int, default=42)
        s.add_argument('--pep-sampling', help='all、non_anchors 或肽内位置列表；配合 --sampling-mode')
    return p


def validate_options(args):
    import importlib.util
    if args.num_templates < 1 or args.num_recycles < 1 or args.batch_size < 1:
        raise ValueError('num-templates、num-recycles、batch-size 必须为正整数')
    allowed = {f'model_{n}_ptm{suffix}' for n in range(1, 6) for suffix in ('', '_ft')}
    if not set(args.models) <= allowed or len(set(args.models)) != len(args.models):
        raise ValueError('models 只支持不重复的 model_1_ptm..model_5_ptm 及其 _ft 形式')
    for name in args.models:
        file = Path(args.fine_tuned_params) if name.endswith('_ft') else Path(args.params_dir) / 'params' / f'params_{name}.npz'
        if not file.is_file():
            raise ValueError(f'模型权重不存在：{file}')
    if args.mode == 'te' and importlib.util.find_spec('modeller') is None:
        raise ValueError('TE 路线需要先安装并授权 Modeller；默认 --mode ig 不需要它')
    if args.pep_sampling and not args.sampling_mode:
        raise ValueError('--pep-sampling 必须配合 --sampling-mode 使用')


def prepare_batch(rows, args, work):
    import pandas as pd
    from .backend import StructureModeler
    frames = []
    for row in rows:
        runner = StructureModeler(
            peptide=row['peptide'], mhc_seq=row['mhc_seq'], mhc_type=row['mhc_type'], id=row['id'],
            anchors=[int(a) for a in row['anchors'].split(';')] if row['anchors'] else None,
            predict_anchor=not bool(row['anchors']), output_dir=str(work),
            num_templates=args.num_templates, best_n_templates=args.num_templates,
            num_recycles=args.num_recycles, models=args.models,
            alphafold_param_folder=args.params_dir, fine_tuned_model_path=args.fine_tuned_params,
            no_modelling=args.mode == 'ig', return_all_outputs=args.return_all_outputs,
            sampling_mode=args.sampling_mode, n_times_sampling=args.n_times_sampling,
            sampling_seed=args.sampling_seed, pep_sampling=args.pep_sampling)
        runner.run_PMGen(run_alphafold=False)
        if not row['anchors']:
            info = json.loads((work / 'pandora' / row['id'] / ('no_modelling_output_dict.json' if args.mode == 'ig' else 'anchors.json')).read_text())
            anchors = info['target_anchors'][0] if args.mode == 'ig' else info['anchors']
            row['anchors'] = ';'.join(map(str, anchors))
        frame = pd.read_table(runner.alphafold_input_file)
        frame['targetid'] = row['id']
        frame['outfile_prefix'] = str(work / 'alphafold' / row['id'] / row['id'])
        frames.append(frame)
    target_file = work / 'targets.tsv'
    pd.concat(frames, ignore_index=True).to_csv(target_file, sep='\t', index=False)
    write_table(work / 'candidates.tsv', rows, FIELDS + ['source_id'])
    return target_file


def af2_command(args, targets):
    params = [args.fine_tuned_params if name.endswith('_ft') else 'classic' for name in args.models]
    command = [sys.executable, str(ROOT / 'AFfine/run_prediction.py'), '--targets', str(targets),
               '--data_dir', args.params_dir, '--model_names', *args.models,
               '--model_params_files', *params, '--ignore_identities',
               '--num_recycles', str(args.num_recycles), '--num_templates', str(args.num_templates)]
    if args.mode == 'te':
        command.append('--no_initial_guess')
    if args.return_all_outputs:
        command.append('--return_all_outputs')
    if args.sampling_mode:
        command += ['--sampling_mode', '--return_all_outputs', '--n_times_sampling', str(args.n_times_sampling),
                    '--sampling_seed', str(args.sampling_seed)]
        if args.pep_sampling:
            command += ['--pep_sampling', args.pep_sampling]
    return command


def run(args):
    rows = read_inputs(args)
    args.params_dir = str(Path(args.params_dir).resolve())
    args.fine_tuned_params = str(Path(args.fine_tuned_params).resolve())
    validate_options(args)
    candidates = expand(rows, args.anchor_mode)
    if args.anchor_mode == 'netmhc':
        for row in candidates:
            key = 'NETMHCPAN_PATH' if row['mhc_type'] == 1 else 'NETMHCIIPAN_PATH'
            if not row['anchors'] and not os.environ.get(key):
                raise ValueError(f'{row["source_id"]}: --anchor-mode netmhc 需要设置 {key}')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # 锁住整个输出目录，防止两个进程交叉覆盖检查点和结果。
    import fcntl
    with (root / '.run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'输出目录已有运行进程：{root}') from None
        return run_locked(args, rows, candidates, root)


def run_locked(args, rows, candidates, root):
    options = {k: v for k, v in vars(args).items() if k not in ('resume', 'command', 'output', 'input', 'mhc_fasta')}
    fingerprint = hashlib.sha256(json.dumps({'rows': rows, 'options': options}, sort_keys=True).encode()).hexdigest()
    manifest_path = root / 'run.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume:
            raise ValueError('输出目录已有任务；请使用新目录，或通过 --resume 继续同一个任务')
        if old['fingerprint'] != fingerprint:
            raise ValueError('输入或模型参数与已有任务不同；请使用新输出目录')
    write_json(manifest_path, dict(fingerprint=fingerprint, options=options, selected_inputs=len(rows),
                                  anchor_candidates=len(candidates), started_at=now()))
    write_table(root / 'input.tsv', rows, FIELDS)
    write_table(root / 'expanded.tsv', candidates, FIELDS + ['source_id'])
    completed = {row['id']: result for row in rows
                 if (result := load_checkpoint(root, row['id'], fingerprint)) is not None}
    if args.command == 'predict':
        write_table(root / 'results.tsv', list(completed.values()), RESULT_FIELDS)
    def progress(status):
        write_json(root / 'progress.json', dict(status=status, selected_inputs=len(rows),
                   completed_inputs=len(completed), remaining_inputs=len(rows) - len(completed),
                   anchor_candidates=len(candidates), updated_at=now(), pid=os.getpid()))

    progress('preparing')
    try:
        # 按原始顺序固定分批，续跑时复用同批目录中的有效产物。
        for offset in range(0, len(rows), args.batch_size):
            original_batch = rows[offset:offset + args.batch_size]
            sources = {r['id'] for r in original_batch if r['id'] not in completed}
            if not sources:
                continue
            group = [r for r in candidates if r['source_id'] in sources]
            work = root / 'work' / f'batch{offset // args.batch_size:05d}'
            work.mkdir(parents=True, exist_ok=True)
            targets = prepare_batch(group, args, work)
            print(f'{now()} 已准备 {len(sources)} 条样本 / {len(group)} 个锚点候选：{targets}', flush=True)
            if args.command == 'prepare':
                continue
            progress('predicting')
            log = root / 'logs' / f'batch{offset // args.batch_size:05d}.log'
            log.parent.mkdir(exist_ok=True)
            command = af2_command(args, targets)
            write_json(work / 'command.json', command)
            with log.open('ab') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            scored = []
            for candidate in group:
                folder = work / 'alphafold' / candidate['id']
                pattern = re.compile(re.escape(candidate['id']) + r'_model_\d+_('
                                     + '|'.join(map(re.escape, args.models)) + r')\.pdb$')
                pdbs = [p for p in folder.glob('*.pdb') if pattern.fullmatch(p.name)]
                if len(pdbs) != len(args.models):
                    raise ValueError(f'{candidate["id"]}: 预测产物数量与模型数量不符；请检查 {log}')
                scored.extend(validate_candidate(candidate, pdb) for pdb in pdbs)
            for row in original_batch:
                if row['id'] in sources:
                    selected = [r for r in scored if r['source_id'] == row['id']]
                    completed[row['id']] = save_best(root, row, selected, fingerprint)
                    print(f'{now()} 已完成 {len(completed)}/{len(rows)}：{row["id"]}', flush=True)
            write_table(root / 'results.tsv', [completed[r['id']] for r in rows if r['id'] in completed], RESULT_FIELDS)
            progress('running')
        progress('prepared' if args.command == 'prepare' else 'complete')
    except BaseException as exc:
        progress('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed')
        write_json(root / 'error.json', dict(error=str(exc), updated_at=now()))
        raise
    print(f'输出目录：{root}', flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'check':
            return subprocess.call([sys.executable, str(ROOT / 'scripts/check_install.py')])
        if args.command == 'af2':
            remainder = args.af2_args[1:] if args.af2_args[:1] == ['--'] else args.af2_args
            return subprocess.call([sys.executable, str(ROOT / 'AFfine/run_prediction.py'), *remainder], cwd=ROOT)
        return run(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

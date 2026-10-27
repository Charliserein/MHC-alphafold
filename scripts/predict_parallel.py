#!/usr/bin/env python3
"""按原样本拆分的多 GPU/多进程预测；每个 worker 使用独立输出目录。"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mhcalphafold.inputs import FIELDS, read_inputs
from mhcalphafold.outputs import RESULT_FIELDS, write_json, write_table


def aggregate(root, rows, workers):
    complete = {}
    for worker in range(workers):
        folder = root / f'worker{worker:02d}'
        path = folder / 'results.tsv'
        if not path.exists():
            continue
        with path.open(newline='') as stream:
            for result in csv.DictReader(stream, delimiter='\t'):
                for key in ('pdb_path', 'plddt_path', 'pae_path', 'ptm_path'):
                    result[key] = f'worker{worker:02d}/' + result[key]
                complete[result['id']] = result
    ordered = [complete[r['id']] for r in rows if r['id'] in complete]
    write_table(root / 'results.tsv', ordered, RESULT_FIELDS)
    return len(ordered)


def run(args, extra):
    if args.workers < 1:
        raise ValueError('workers 必须为正整数')
    forbidden = {'--input', '--output', '--peptide', '--mhc-seq', '--mhc-fasta', '--id', '--resume'}
    if any(item.split('=')[0] in forbidden for item in extra):
        raise ValueError('worker 参数不能覆盖 input/output/resume 或单条样本参数')
    gpus = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if not gpus:
        raise ValueError('请通过 --gpus 指定 GPU，例如 1,0')
    rows = read_inputs(argparse.Namespace(input=args.input, peptide=None, mhc_seq=None, mhc_fasta=None, anchors=None))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.parallel.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(workers=args.workers, gpus=gpus, extra=extra, inputs=rows)
    manifest = root / 'parallel.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != config:
            raise ValueError('已有并行任务；续跑需使用 --resume 并保持输入、worker 数和参数一致')
    write_json(manifest, config)
    write_table(root / 'input.tsv', rows, FIELDS)
    processes = []
    stopped = False
    worker_failed = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        for process, handle in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for worker in range(min(args.workers, len(rows))):
            part = root / f'input_worker{worker:02d}.tsv'
            write_table(part, rows[worker::args.workers], FIELDS)
            output = root / f'worker{worker:02d}'
            command = [str(ROOT / 'mhcalphafold.sh'), 'predict', '--input', str(part), '--output', str(output), *extra]
            if args.resume:
                command.append('--resume')
            handle = (root / f'worker{worker:02d}.log').open('ab')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[worker % len(gpus)])
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, handle))
        while any(process.poll() is None for process, _ in processes):
            completed = aggregate(root, rows, args.workers)
            failed = [p.returncode for p, _ in processes if p.poll() not in (None, 0)]
            write_json(root / 'progress.json', dict(status='running', completed_inputs=completed,
                       selected_inputs=len(rows), active_workers=sum(p.poll() is None for p, _ in processes),
                       failed_workers=len(failed), gpus=gpus))
            if failed or stopped:
                worker_failed = bool(failed)
                stop(None, None)
                break
            time.sleep(5)
    finally:
        for process, handle in processes:
            process.wait()
            handle.close()
    completed = aggregate(root, rows, args.workers)
    passed = not stopped and all(p.returncode == 0 for p, _ in processes) and completed == len(rows)
    write_json(root / 'progress.json', dict(status='complete' if passed else 'failed' if worker_failed else 'interrupted' if stopped else 'failed',
               completed_inputs=completed, selected_inputs=len(rows), workers=args.workers, gpus=gpus))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--resume', action='store_true')
    args, extra = parser.parse_known_args()
    try:
        raise SystemExit(run(args, extra))
    except (ValueError, BlockingIOError) as exc:
        parser.error(str(exc))

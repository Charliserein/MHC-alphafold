#!/usr/bin/env python3
"""检查实际导入路径、权重、模板、工具和 JAX GPU。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'PANDORA'), str(ROOT / 'AFfine'), str(ROOT)]


def check():
    import jax
    import haiku
    import numpy as np
    import tensorflow as tf
    import PANDORA
    from PANDORA import Database
    from alphafold.model import modules
    db = Database.load()
    checks = {'python': sys.executable, 'python_base': sys.base_prefix,
              'jax': jax.__version__, 'haiku': haiku.__version__, 'tensorflow': tf.__version__,
              'numpy': np.__version__, 'pandora_source': PANDORA.__file__,
              'af2_source': modules.__file__, 'database': PANDORA.PANDORA_data,
              'templates_mhc1': len(db.MHCI_data), 'templates_mhc2': len(db.MHCII_data),
              'devices': [str(device) for device in jax.devices()],
              'gpu_available': any(device.platform == 'gpu' for device in jax.devices()),
              'optional_modeller_installed': importlib.util.find_spec('modeller') is not None,
              'optional_netmhcpan_configured': bool(os.environ.get('NETMHCPAN_PATH')),
              'optional_netmhciipan_configured': bool(os.environ.get('NETMHCIIPAN_PATH'))}
    for name, args in [('blastp', ['-version']), ('muscle', ['-version'])]:
        result = subprocess.run([name, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        checks[name] = result.stdout.strip()
    weights = ROOT / 'AFfine/af_params/params_original/params/params_model_2_ptm.npz'
    checks['default_weights_present'] = weights.is_file()
    checks['local_sources'] = all(Path(checks[key]).resolve().is_relative_to(ROOT)
                                  for key in ('pandora_source', 'af2_source'))
    checks['passed'] = bool(checks['local_sources'] and checks['gpu_available']
                            and checks['default_weights_present'] and len(db.MHCI_data) and len(db.MHCII_data))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    validation = ROOT / 'validation'
    validation.mkdir(exist_ok=True)
    (validation / 'install-check.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    return 0 if checks['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(check())

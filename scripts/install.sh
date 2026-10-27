#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --check-only ]]; then
    exec "$project_root/mhcalphafold.sh" check
fi
bootstrap_python="${MHC_AF_BOOTSTRAP_PYTHON:-python3.10}"
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    "$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "需要 Python 3.10"'
    "$bootstrap_python" -m venv "$project_root/.venv"
fi
"$project_root/.venv/bin/python" -m pip install --upgrade 'pip==24.3.1' 'setuptools==75.8.0' 'wheel==0.45.1'
"$project_root/.venv/bin/python" -m pip install -r "$project_root/requirements.txt" -r "$project_root/requirements-cuda.txt"
"$project_root/.venv/bin/python" -m pip check
exec "$project_root/mhcalphafold.sh" check

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/activate.sh"
if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    echo '缺少独立 Python 环境，请先执行 bash scripts/install.sh' >&2
    exit 2
fi
exec "$VIRTUAL_ENV/bin/python" "$MHC_AF_ROOT/predict.py" "$@"

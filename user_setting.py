"""可迁移的项目路径与可选授权工具配置。"""
import os
from pathlib import Path

pmgen_abs_dir = str(Path(__file__).resolve().parent)
netmhcipan_path = os.environ.get('NETMHCPAN_PATH', '')
netmhciipan_path = os.environ.get('NETMHCIIPAN_PATH', '')

#!/usr/bin/env python3
"""从 PMGen 上游资源归档恢复模板库和两套权重；支持本地 ZIP。"""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    'database': ('https://owncloud.gwdg.de/index.php/s/gebWa9TN3VNHy31/download',
                 'bf244d9b0bc6be472e3cbcde3c90485773765ab80caf12140ac35afc0af1d5e2', ROOT / 'resources/pandora_db'),
    'weights': ('https://owncloud.gwdg.de/index.php/s/M1YQOgKxLbVjO0G/download',
                '73f67d7b4d643cf5685db098912a20752fab0495e5a1ff92ad5aef010e52efdf', ROOT / 'AFfine')}


def install(name, supplied, temp):
    url, expected, destination = ARCHIVES[name]
    archive = Path(supplied).resolve() if supplied else temp / f'{name}.zip'
    if not supplied:
        print(f'下载 {name}: {url}', flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, archive.open('wb') as output:
            while chunk := source.read(4 * 1024 * 1024):
                output.write(chunk)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'{name}: SHA256 不匹配，上游归档可能已变化；未解压')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError(f'{name}: ZIP CRC 校验失败')
        for member in zipped.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('ZIP 含越界路径')
        zipped.extractall(destination)
    print(f'{name} 已安装到 {destination}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database-zip')
    p.add_argument('--weights-zip')
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='mhcalphafold-') as temp:
        install('database', args.database_zip, Path(temp))
        install('weights', args.weights_zip, Path(temp))

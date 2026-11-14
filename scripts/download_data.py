"""Download the official ARNOR 2.0 archive; verify and extract only known files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path('data/arnor'))
    p.add_argument('--manifest',type=Path,default=Path('configs/data_manifest.json'))
    a=p.parse_args(); manifest=json.loads(a.manifest.read_text())
    if a.out.exists(): raise FileExistsError('Use a new output directory to preserve existing data')
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'arnor.tar.gz'
        urllib.request.urlretrieve(manifest['download'],archive)
        extracted={}
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                name=Path(member.name).name
                if name not in manifest['files']: continue
                if not member.isfile() or name in extracted: raise ValueError('Unexpected archive member')
                content=tar.extractfile(member).read()
                expected=manifest['files'][name]
                if len(content)!=expected['bytes'] or hashlib.sha256(content).hexdigest()!=expected['sha256']:
                    raise ValueError(f'Official data changed: {name}; do not silently mix versions')
                extracted[name]=content
        if set(extracted)!=set(manifest['files']): raise ValueError('Missing expected data files')
        a.out.mkdir(parents=True)
        for name,content in extracted.items(): (a.out/name).write_bytes(content)
    print('Verified all four original files; no raw data should be committed.')

if __name__=='__main__': main()

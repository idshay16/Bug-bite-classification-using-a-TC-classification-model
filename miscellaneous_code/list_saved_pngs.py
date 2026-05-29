#!/usr/bin/env python3
import os, json, sys
from pathlib import Path
ROOT = Path('cyclone_dataset')
if not ROOT.exists():
    print('No cyclone_dataset folder found')
    sys.exit(0)

for p in sorted(ROOT.iterdir()):
    if not p.is_dir() or p.name == '_tmp_nc':
        continue
    pngs = list(p.rglob('*.png'))
    print(f"{p.name}: {len(pngs)}")
    for pp in sorted(pngs, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        print('  ', pp.name)

dl = ROOT / 'download_log.json'
if dl.exists():
    try:
        j = json.loads(dl.read_text())
        done = j.get('done')
        if isinstance(done, list):
            print(f"\ndownload_log.json: done entries = {len(done)}")
            for item in done[:20]:
                print('  ', item)
        else:
            print('\ndownload_log.json: unexpected structure')
    except Exception as e:
        print('\ndownload_log.json read error:', e)
else:
    print('\ndownload_log.json: missing')

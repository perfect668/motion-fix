"""Offline scene collision manifest builder.

CoACD is optional.  Without it, provide pre-decomposed convex pieces with
``--pieces``; the script only writes a deterministic manifest.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mesh',required=True,type=Path); p.add_argument('--output',required=True,type=Path); p.add_argument('--pieces',nargs='*',default=[]); a=p.parse_args()
    if not a.pieces:
        raise SystemExit('CoACD preprocessing is optional; provide --pieces piece_000.obj ...')
    a.output.mkdir(parents=True,exist_ok=True)
    pieces=[]
    for item in a.pieces:
        src=Path(item); dst=a.output/src.name; dst.write_bytes(src.read_bytes()); pieces.append(dst.name)
    (a.output/'collision_manifest.json').write_text(json.dumps({'object_id':a.mesh.stem,'source_mesh':str(a.mesh),'pieces':pieces},indent=2))
    print(a.output/'collision_manifest.json')
if __name__=='__main__': main()

import glob
import json
import os

jsons = sorted(glob.glob('Real-Data/T12/T12-B6Aggolmeration/*.json'))
print(f'Tracing sticking pellets across {len(jsons)} labeled frames:\n')
for jf in jsons:
    with open(jf) as fp:
        d = json.load(fp)
    fname = os.path.basename(jf).replace('.json', '')
    shapes = d.get('shapes', [])
    obj_strs = []
    for s in shapes:
        (x1, y1), (x2, y2) = s['points']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = abs(x2 - x1), abs(y2 - y1)
        obj_strs.append(f"({cx:.0f},{cy:.0f}, {w:.0f}x{h:.0f})")
    print(f"{fname}: {len(shapes)} sticking -> " + "; ".join(obj_strs))

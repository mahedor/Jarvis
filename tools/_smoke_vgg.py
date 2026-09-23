"""Throwaway smoke test for the vgg-face encoder. Delete after verifying."""
import sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from face_utils import load_encoder, load_reference_crops
from recognition_metrics import l2_normalize

REF = HERE.parent / "data" / "reference_faces"

persons = load_reference_crops(REF)
print("persons:", {k: len(v) for k, v in persons.items()})

t0 = time.perf_counter()
enc = load_encoder("vgg-face")
print(f"built in {time.perf_counter() - t0:.1f}s")

names = sorted(persons)[:2]
vecs = {}
for n in names:
    got = []
    for fname, img in persons[n][:3]:
        t = time.perf_counter()
        v = enc.embed(img)
        got.append(v)
        print(f"  {n}/{fname}: dim={v.shape} dtype={v.dtype} "
              f"norm={np.linalg.norm(v):.4f} {(time.perf_counter()-t)*1000:.0f}ms")
    vecs[n] = l2_normalize(np.vstack(got))

a, b = names
print("same-person cos :", float((vecs[a] @ vecs[a].T)[0, 1]))
print("cross-person cos:", float((vecs[a] @ vecs[b].T)[0, 0]))
print("fallbacks:", enc.fallback_count)

# empty-crop contract
try:
    enc.embed(np.empty((0, 0, 3), dtype=np.uint8))
    print("EMPTY CROP: no raise -- BUG")
except ValueError as e:
    print("empty crop raises ValueError as contracted:", e)

#!/usr/bin/env python3
"""
preview_hyper_mesh.py — decode assets/js/hyper-mesh.js and render it, so you
can eyeball that the cottage geometry the hypercentric engine will draw is
intact.  Pure stdlib: no numpy/PIL — it emits an SVG (open in any browser).

  * integrity: buffer byte-lengths vs vertex/triangle counts, index bounds,
    NaN check, recomputed radius vs the stored one.
  * render: two 3/4 views, painter's-algorithm z-sort, flat Lambert shading
    in the site accent color.  Writes scripts/hyper-mesh-preview.svg.
"""

import base64
import math
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESH = ROOT / "assets" / "js" / "hyper-mesh.js"
OUT = Path(__file__).resolve().parent / "hyper-mesh-preview.svg"

ACCENT = (0xC4, 0x55, 0x3B)   # warm terracotta, reads as a cottage
BG = "#0d0c0a"


def extract(src, key):
    m = re.search(key + r'\s*:\s*"([^"]*)"', src)
    return base64.b64decode(m.group(1))


def load():
    src = MESH.read_text()
    pos = list(struct.unpack("<%df" % (len(b := extract(src, "positions")) // 4), b))
    nrm = list(struct.unpack("<%df" % (len(b := extract(src, "normals")) // 4), b))
    idx = list(struct.unpack("<%dI" % (len(b := extract(src, "indices")) // 4), b))
    vc = int(re.search(r"vertexCount:\s*(\d+)", src).group(1))
    tc = int(re.search(r"triangleCount:\s*(\d+)", src).group(1))
    rad = float(re.search(r"radius:\s*([\d.eE+-]+)", src).group(1))
    return pos, nrm, idx, vc, tc, rad


def integrity(pos, nrm, idx, vc, tc, rad):
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    check("positions length", len(pos) == vc * 3, f"{len(pos)} == {vc}*3")
    check("normals length", len(nrm) == vc * 3, f"{len(nrm)} == {vc}*3")
    check("indices length", len(idx) == tc * 3, f"{len(idx)} == {tc}*3")
    check("index bounds", all(0 <= i < vc for i in idx), f"max={max(idx)} < {vc}")
    check("no NaN/Inf in positions", all(math.isfinite(v) for v in pos))
    r = max(math.sqrt(pos[i]**2 + pos[i+1]**2 + pos[i+2]**2) for i in range(0, len(pos), 3))
    check("recomputed radius ~= stored", abs(r - rad) < 1e-3, f"{r:.4f} vs {rad:.4f}")
    cen = [sum(pos[i::3]) / vc for i in range(3)]
    check("roughly centered", all(abs(c) < rad for c in cen),
          f"centroid=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f})")
    return ok


def rot(p, ax, ay):
    x, y, z = p
    ca, sa = math.cos(ay), math.sin(ay)          # yaw
    x, z = ca * x + sa * z, -sa * x + ca * z
    cb, sb = math.cos(ax), math.sin(ax)          # pitch
    y, z = cb * y - sb * z, sb * y + cb * z
    return x, y, z


def view_svg(pos, idx, rad, ax, ay, W, H, ox):
    """One orthographic 3/4 view, returned as a list of <polygon> strings."""
    s = 0.42 * min(W, H) / rad
    cx, cy = ox + W / 2, H / 2
    light = (-0.4, -0.7, 0.6)
    lm = math.sqrt(sum(c*c for c in light)); light = tuple(c/lm for c in light)

    tris = []
    for t in range(0, len(idx), 3):
        vs = [pos[idx[t+k]*3: idx[t+k]*3+3] for k in range(3)]
        r = [rot(v, ax, ay) for v in vs]
        # face normal in view space
        u = [r[1][i]-r[0][i] for i in range(3)]
        w = [r[2][i]-r[0][i] for i in range(3)]
        n = (u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0])
        nl = math.sqrt(sum(c*c for c in n)) or 1
        n = tuple(c/nl for c in n)
        if n[2] <= 0:      # back-face cull (view looks down -z after projection)
            pass
        shade = max(0.12, sum(n[i]*light[i] for i in range(3)))
        shade = 0.15 + 0.85 * abs(shade)
        col = "#%02x%02x%02x" % tuple(min(255, int(c*shade)) for c in ACCENT)
        zavg = sum(v[2] for v in r) / 3
        pts = " ".join(f"{cx + p[0]*s:.1f},{cy - p[1]*s:.1f}" for p in r)
        tris.append((zavg, f'<polygon points="{pts}" fill="{col}" stroke="{col}" stroke-width="0.3"/>'))
    tris.sort(key=lambda a: a[0])          # painter's: far first
    return [p for _, p in tris]


def main():
    pos, nrm, idx, vc, tc, rad = load()
    print(f"Decoded {MESH.name}: {vc} verts, {tc} tris, radius {rad:.4f}")
    ok = integrity(pos, nrm, idx, vc, tc, rad)

    W, H = 460, 460
    polysA = view_svg(pos, idx, rad, math.radians(20), math.radians(-35), W, H, 0)
    polysB = view_svg(pos, idx, rad, math.radians(20), math.radians(55), W, H, W)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*W}" height="{H}" '
           f'viewBox="0 0 {2*W} {H}">',
           f'<rect width="{2*W}" height="{H}" fill="{BG}"/>']
    svg += polysA + polysB
    svg += [f'<text x="12" y="{H-14}" fill="#7c766a" font-family="sans-serif" '
            f'font-size="13">Cottage_FREE.obj -> HYPER_MESH  ·  {vc} verts / {tc} tris  ·  r={rad:.2f}</text>',
            '</svg>']
    OUT.write_text("\n".join(svg))
    print(f"\nWrote preview: {OUT}")
    print("Integrity:", "ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***")


if __name__ == "__main__":
    main()

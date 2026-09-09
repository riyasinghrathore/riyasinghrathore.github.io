#!/usr/bin/env python3
"""
obj_to_hyper_mesh.py — convert a Wavefront .obj into the window.HYPER_MESH
payload that assets/js/hyper-engine.js consumes.

Output format (assets/js/hyper-mesh.js):

    window.HYPER_MESH = {
      positions: "<base64 little-endian Float32, xyz flat>",
      normals:   "<base64 little-endian Float32, xyz flat>",
      indices:   "<base64 little-endian Uint32, triangle list>",
      vertexCount, triangleCount, radius
    };

The engine wants one interleaved index buffer, but OBJ indexes position and
normal separately, so we weld each unique (v, vn) pair into a single vertex.
Quads (and any n-gon) are fan-triangulated.  Faces with no normal get a
generated flat normal.  The mesh is centered on its bounding-box center and
`radius` is the max vertex distance from that center (what the engine uses to
frame the scene).

    python3 scripts/obj_to_hyper_mesh.py <input.obj> [output.js]
"""

import base64
import struct
import sys
from pathlib import Path


def parse_obj(path):
    verts = []          # list of (x,y,z)
    norms = []          # list of (x,y,z)
    faces = []          # list of list of (vi, ni-or-None), 0-based
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vn "):
                _, x, y, z = line.split()[:4]
                norms.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face = []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    vi = int(parts[0])
                    ni = int(parts[2]) if len(parts) >= 3 and parts[2] else None
                    vi = vi - 1 if vi > 0 else len(verts) + vi
                    if ni is not None:
                        ni = ni - 1 if ni > 0 else len(norms) + ni
                    face.append((vi, ni))
                faces.append(face)
    return verts, norms, faces


def face_normal(verts, face):
    a = verts[face[0][0]]
    b = verts[face[1][0]]
    c = verts[face[2][0]]
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (u[1] * v[2] - u[2] * v[1],
         u[2] * v[0] - u[0] * v[2],
         u[0] * v[1] - u[1] * v[0])
    m = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
    return (n[0] / m, n[1] / m, n[2] / m)


def build(verts, norms, faces):
    out_pos = []        # flat xyz
    out_nrm = []        # flat xyz
    out_idx = []
    weld = {}           # (vi, ni) -> new index

    def emit(vi, ni, fallback_n):
        key = (vi, ni)
        idx = weld.get(key)
        if idx is None:
            idx = len(out_pos) // 3
            weld[key] = idx
            out_pos.extend(verts[vi])
            n = norms[ni] if ni is not None else fallback_n
            out_nrm.extend(n)
        return idx

    for face in faces:
        fn = face_normal(verts, face) if len(face) >= 3 else (0.0, 1.0, 0.0)
        # fan-triangulate
        for k in range(1, len(face) - 1):
            for (vi, ni) in (face[0], face[k], face[k + 1]):
                out_idx.append(emit(vi, ni, fn))
    return out_pos, out_nrm, out_idx


def center_and_radius(pos):
    xs = pos[0::3]; ys = pos[1::3]; zs = pos[2::3]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    r = 0.0
    for i in range(0, len(pos), 3):
        pos[i] -= cx; pos[i + 1] -= cy; pos[i + 2] -= cz
        d = (pos[i] ** 2 + pos[i + 1] ** 2 + pos[i + 2] ** 2) ** 0.5
        if d > r:
            r = d
    return r


def b64f32(flat):
    return base64.b64encode(struct.pack("<%df" % len(flat), *flat)).decode()


def b64u32(flat):
    return base64.b64encode(struct.pack("<%dI" % len(flat), *flat)).decode()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parent.parent / "assets" / "js" / "hyper-mesh.js"

    verts, norms, faces = parse_obj(src)
    pos, nrm, idx = build(verts, norms, faces)
    radius = center_and_radius(pos)
    vcount = len(pos) // 3
    tcount = len(idx) // 3

    js = (
        f"// Auto-generated from {src.name} — indexed geometry, base64-encoded little-endian typed arrays.\n"
        "// positions/normals: Float32 xyz flat; indices: Uint32 triangle list. Mesh is centered at origin.\n"
        "window.HYPER_MESH = {\n"
        f'  positions: "{b64f32(pos)}",\n'
        f'  normals:   "{b64f32(nrm)}",\n'
        f'  indices:   "{b64u32(idx)}",\n'
        f"  vertexCount: {vcount},\n"
        f"  triangleCount: {tcount},\n"
        f"  radius: {radius}\n"
        "};\n"
    )
    dst.write_text(js)
    print(f"wrote {dst}")
    print(f"  source verts={len(verts)} norms={len(norms)} faces={len(faces)}")
    print(f"  welded vertices={vcount} triangles={tcount} radius={radius:.4f}")


if __name__ == "__main__":
    main()

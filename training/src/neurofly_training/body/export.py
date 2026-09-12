"""The fly body for other renderers: a glTF of the meshes, and poses per frame.

``export_body`` writes the MuJoCo fly as a ``.glb``: one node per body, named as in
the model, all children of the root (a flat hierarchy), each carrying its geoms as
mesh children with their local transforms. A renderer then animates the fly by
setting every body node's world position and orientation from a pose stream.

``PoseRecorder`` collects those world poses (position and quaternion per body)
from the simulation while it runs, and writes ``poses.json``:

    {"bodies": [...names], "fps": 50, "frames": [[x, y, z, qw, qx, qy, qz] * n_bodies, ...]}

Units are centimetres; MuJoCo is z-up (the viewer rotates to y-up).
"""
from __future__ import annotations

import json

import numpy as np

MESH, BOX, SPHERE, CAPSULE, CYLINDER, ELLIPSOID = 7, 6, 2, 3, 5, 4


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _transform(pos, mat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _geom_mesh(m, g):
    """A trimesh for geom ``g`` in its own frame, or None for planes and unknown types."""
    import trimesh
    t = int(m.geom_type[g])
    size = np.asarray(m.geom_size[g])
    if t == MESH:
        did = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
        return trimesh.Trimesh(vertices=np.asarray(m.mesh_vert[va:va + vn]),
                               faces=np.asarray(m.mesh_face[fa:fa + fn]), process=False)
    if t == BOX:
        return trimesh.creation.box(extents=2 * size[:3])
    if t == SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if t == CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
    if t == CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
    if t == ELLIPSOID:
        s = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        s.apply_scale(size[:3])
        return s
    return None


def export_body(physics, out: str, skip_bodies=("world",)) -> dict:
    """Write the body model as a .glb; returns a summary."""
    import trimesh
    m, d = physics.model, physics.data
    scene = trimesh.Scene()
    names = [m.id2name(i, "body") or f"body{i}" for i in range(m.nbody)]
    n_geoms = 0
    for b in range(m.nbody):
        if names[b] in skip_bodies:
            continue
        T_body = _transform(d.xpos[b], d.xmat[b])
        scene.graph.update(frame_to=names[b], frame_from=scene.graph.base_frame, matrix=T_body)
        for g in np.flatnonzero(np.asarray(m.geom_bodyid) == b):
            mesh = _geom_mesh(m, int(g))
            if mesh is None:
                continue
            rgba = np.asarray(m.geom_rgba[g])
            colour = (rgba * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=colour)
            T_geom = _transform(d.geom_xpos[g], d.geom_xmat[g])
            local = np.linalg.inv(T_body) @ T_geom
            # geoms often share their body's name in MJCF; keep node names unique
            gname = f"{m.id2name(int(g), 'geom') or 'geom'}#g{int(g)}"
            scene.add_geometry(mesh, node_name=gname, geom_name=gname, parent_node_name=names[b],
                               transform=local)
            n_geoms += 1
    scene.export(out)
    return {"path": out, "bodies": [n for n in names if n not in skip_bodies], "geoms": n_geoms}


class PoseRecorder:
    """Collects world poses of every body from ``physics`` on each ``record()``."""

    def __init__(self, physics, fps: float, skip_bodies=("world",)):
        """``physics`` may be a Physics or a callable returning the current one (dm_control
        environments rebuild their physics on some resets)."""
        self._get = physics if callable(physics) else (lambda: physics)
        m = self._get().model
        self.ids = [i for i in range(m.nbody)
                    if (m.id2name(i, "body") or f"body{i}") not in skip_bodies]
        self.bodies = [m.id2name(i, "body") or f"body{i}" for i in self.ids]
        self.fps = float(fps)
        self.frames: list[list[float]] = []

    def record(self) -> None:
        d = self._get().data
        row = []
        for i in self.ids:
            row += [float(v) for v in d.xpos[i]] + [float(v) for v in d.xquat[i]]
        self.frames.append(row)

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump({"bodies": self.bodies, "fps": self.fps, "units": "cm", "up": "z",
                       "frames": self.frames}, f)
        return path

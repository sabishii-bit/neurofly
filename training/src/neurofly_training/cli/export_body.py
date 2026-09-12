"""Export the fly body as a glTF for Three.js, Unity, Blender, or any other renderer.

    neurofly export-body --out assets/fly.glb
    neurofly watch runs/<run> --poses assets/poses.json      # then see examples/three_viewer.html

The .glb has one named node per MuJoCo body, all under the root, each carrying its meshes.
A pose stream (world position and quaternion per body per frame, from `watch --poses`)
drives it; examples/three_viewer.html plays one back.
"""
from __future__ import annotations

import argparse
import os

from neurofly_training.body.export import export_body
from neurofly_training.envs import make_body_env


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/fly.glb")
    p.add_argument("--task", default="forward", help="body task whose model to export")
    p.add_argument("--mjcf", default=None, metavar="DIR",
                   help="also write the complete MuJoCo model (fly.xml + meshes) here, so any "
                        "MuJoCo build (C, Unity, WebAssembly) can simulate it")
    args = p.parse_args()
    env = make_body_env(args.task, seed=0)
    env.reset()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    info = export_body(env.physics, args.out)
    if args.mjcf:
        from dm_control import mjcf
        os.makedirs(args.mjcf, exist_ok=True)
        mjcf.export_with_assets(env.dm_env.task.root_entity.mjcf_model, args.mjcf,
                                out_file_name="fly.xml")
        n = len(os.listdir(args.mjcf))
        print(f"wrote MuJoCo model to {args.mjcf}/fly.xml with {n - 1} asset files")
    env.close()
    print(f"wrote {info['path']}: {len(info['bodies'])} bodies, {info['geoms']} geoms "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("bodies:", ", ".join(info["bodies"][:12]), "...")


if __name__ == "__main__":
    main()

"""Every third-party module the code imports must be declared by its package.

This is the test that would have caught trimesh and websockets being used without
being listed: a fresh `pip install` would then fail at runtime, not at install time.
"""
import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import name -> distribution name, where they differ
DIST = {"PIL": "pillow", "cv2": "opencv-python", "grpc": "grpcio", "yaml": "pyyaml",
        "dm_control": "flybody", "mujoco": "flybody", "stable_baselines3": "stable-baselines3",
        "neuprint": "neuprint-python", "skimage": "scikit-image", "google": "grpcio"}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _declared(pyproject: str) -> set[str]:
    text = open(pyproject, encoding="utf-8").read()
    names = set()
    for m in re.finditer(r'"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(?:[=<>!~@ ].*)?"', text):
        names.add(m.group(1).lower().replace("_", "-"))
    return names


def _imports(src_dir: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.endswith(".py") or "_pb2" in f:
                continue
            path = os.path.join(root, f)
            tree = ast.parse(open(path, encoding="utf-8").read(), path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    found.setdefault(name.split(".")[0], set()).add(os.path.relpath(path, ROOT))
    return found


@pytest.mark.parametrize("package,own", [("core", {"neurofly_core"}),
                                         ("training", {"neurofly_training", "neurofly_core"})])
def test_imports_are_declared(package, own):
    declared = _declared(os.path.join(ROOT, package, "pyproject.toml"))
    if package == "training":
        declared |= _declared(os.path.join(ROOT, "core", "pyproject.toml"))
    missing = {}
    for mod, files in _imports(os.path.join(ROOT, package, "src")).items():
        if mod in STDLIB or mod in own:
            continue
        dist = DIST.get(mod, mod).lower().replace("_", "-")
        if dist not in declared:
            missing[mod] = sorted(files)
    assert not missing, f"imported but not declared in {package}/pyproject.toml: {missing}"

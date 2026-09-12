# Installation

Two Python packages live in this repository: `neurofly-core` (the runtime, in `core/`) and
`neurofly-training` (everything that builds and trains, in `training/`). Consumers in other
languages need only the runtime, and only until a native core replaces it; developers and
trainers need both.

## Requirements

| What | Needed for | Notes |
|---|---|---|
| Python 3.10 or newer | everything | 3.10 is what the shared environment uses; 3.11 and 3.12 work |
| about 1.5 GB of disk | the connectome files and their cache | in `data/malecns/` by default |
| a C compiler is **not** needed | | every dependency ships wheels; numba brings its own JIT |
| a CUDA build of torch | optional | the full brain on a GPU; CPU runs everything, slower |
| ViGEmBus driver (Windows) | optional | a virtual Xbox controller for `--pad-buttons` / `--axes` |
| Tesseract (any platform) | optional | reading numbers off the screen for reward |
| Node 18+, Rust, Go 1.21+ | optional | only for the respective binding |

Window lookup by title, the XInput controller reader and the virtual gamepad are
Windows-only; screen regions, keyboard, mouse and everything else work on Linux and macOS.

## The Python packages

From a clone of the repository, in a fresh virtual environment:

```powershell
git clone https://github.com/sabishii-bit/neurofly
cd neurofly
python -m venv .venv
.venv\Scripts\Activate.ps1                 # Linux / macOS: source .venv/bin/activate
pip install -e core[pc] -e training[dev]
```

This installs both packages editable (changes to the source take effect without
reinstalling) and puts two commands on the path: `neurofly` and `neurofly-core`.
`python -m neurofly_training` and `python -m neurofly_core` are the same things.

### Extras

Extras are optional groups of dependencies; add them in the brackets, comma-separated.

| Package | Extra | Installs | Enables |
|---|---|---|---|
| core | `pc` | mss, pynput, sounddevice, soundfile, soundcard | screen and sound capture, keyboard and mouse: `run`, `play`, `record` |
| core | `ws` | websockets | `neurofly-core serve --ws` |
| core | `grpc` | grpcio | `neurofly-core serve --grpc` |
| core | `gamepad` | vgamepad | the virtual controller (plus the ViGEmBus driver) |
| core | `video` | imageio, imageio-ffmpeg | reading video files without the training package |
| training | `flybody` | the MuJoCo fruit fly, from GitHub | body tasks: `--task forward`, `--task ball`, `export-body` |
| training | `ocr` | pytesseract | `NumberOnScreen` and `read_number` (plus a Tesseract install) |
| training | `dev` | pytest, pytest-cov, ruff, grpcio-tools | tests, coverage, lint, regenerating the gRPC stubs |

A runtime-only machine: `pip install -e core[pc,grpc]`. A full development machine:
`pip install -e core[pc,ws,grpc,gamepad] -e training[flybody,ocr,dev]`.

### The body: flybody and MuJoCo

`flybody` is not on PyPI; the `flybody` extra pulls it from GitHub and it brings `mujoco`
and `dm_control` with it. It pins `numpy` to 1.26, which is why the training package pins
the same. Without the extra, PC tasks, artifacts, bindings and everything else work; only
the body tasks and `export-body` need it.

The checkout as it stands on the original machine shares the environment of a `flybody`
checkout next door (`..\flybody\.venv`); a fresh clone should make its own environment as
above.

### GPU

The default `torch` wheel is CPU-only. For CUDA, install the build for your driver before
the packages, following the selector on the PyTorch site, for example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then pass `--device cuda` to the commands. The event-driven brain backend is CPU-only; on
CUDA the runtime uses the sparse-product backend automatically.

## The connectome

```powershell
neurofly download
```

Three public files, about 570 MB, into `data/malecns/`. The first load builds a cached
weight matrix in `data/malecns/cache/` (about a minute); later loads take a second. Set
`NEUROFLY_DATA` to keep the data elsewhere:

```powershell
$env:NEUROFLY_DATA = "D:\connectomes\malecns"
```

Nothing else in the project touches the network.

## Check the install

```powershell
python -m pytest -q                # about 45 seconds; body and real-data tests skip if their inputs are missing
neurofly list                      # body tasks, PC tasks, connectome subsets
neurofly inspect --subset central  # the brain and its populations
neurofly bench --subset central    # how fast it steps on this machine
neurofly devices                   # sound inputs the capture can open
```

## The bindings

Each binding is a thin client that starts `neurofly-core serve`, so the runtime must be
installed in a Python on the `PATH` (or an interpreter you name when constructing the
client). None of them is published to a registry yet; use them from the repository.

| Binding | Install and build | Use from your project |
|---|---|---|
| Node (TypeScript) | `cd bindings/node && npm install && npm run build` | `"neurofly": "file:../neurofly/bindings/node"` in `package.json` |
| Rust | nothing to build ahead of time | `neurofly = { path = "../neurofly/bindings/rust" }` in `Cargo.toml` |
| Go | nothing to build ahead of time | `require neurofly v0.0.0` plus `replace neurofly => ../neurofly/bindings/go` |

Smoke tests, against any artifact:

```powershell
node bindings/node/test.js artifacts/myapp
cd bindings/rust && cargo check
cd bindings/go && go run ./example ../../artifacts/base ../../artifacts/from_go
```

A client in any other language needs no package: speak the protocol in
[artifact/SPEC.md](../artifact/SPEC.md) over stdio, WebSocket or gRPC. For gRPC, generate a
stub from `core/src/neurofly_core/rpc/neurofly.proto` with `protoc`.

## Regenerating the gRPC stubs

Only after editing the `.proto`; the generated Python files are committed:

```powershell
cd core/src/neurofly_core/rpc
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. neurofly.proto
```

then change the import at the top of `neurofly_pb2_grpc.py` to
`from neurofly_core.rpc import neurofly_pb2 as neurofly__pb2`.

## Upgrading and removing

Editable installs follow the checkout: `git pull` is the upgrade, except when
`pyproject.toml` gained a dependency, in which case rerun the `pip install -e ...` line.
`pip uninstall neurofly-core neurofly-training` removes the packages; the data, runs and
artifacts directories are yours to delete.

## Known install problems

* **`No module named 'neurofly_core'` after installing only training**: the training package
  depends on `neurofly-core` by name, which resolves only once it is published; from a
  checkout install both with one `pip install -e core -e training` command.
* **`ImportError: numpy` version conflicts**: `flybody` pins numpy 1.26; install the packages
  into a fresh environment rather than on top of another project's.
* **`vgamepad` imports but creating a controller fails**: the ViGEmBus driver is not
  installed; it is a separate download from the ViGEm project.
* **`read_number` always returns `None`**: `pytesseract` is installed but Tesseract itself is
  not on the `PATH`.
* **`numba` compiles on every start**: the cache lives next to the package under
  `__pycache__`; a read-only install location disables it. First start takes a few seconds
  either way.

# Getting started

## Requirements

* Python 3.10 or newer, on Windows, Linux or macOS. Everything runs on CPU; a CUDA build of
  torch is optional.
* About 1.5 GB of disk for the connectome files and their cache.
* For the PC interface: a screen to capture and, optionally, a sound device. Window lookup
  by title is Windows-only; other platforms capture a screen region instead.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e core[pc] -e training[dev]
```

That is the whole install for the PC side. Add `training[flybody]` for the simulated body,
and see [Installation](installation.md) for every extra, GPU support, the bindings and the
problems people hit. It puts two commands on the path: `neurofly` (training) and
`neurofly-core` (runtime).

## Fetch the connectome

```powershell
neurofly download
```

This downloads three public files (about 570 MB) into `data/malecns/`: neuron annotations,
neurotransmitter predictions, and the synapse table. Files already present are skipped. The
first load builds a cached weight matrix in `data/malecns/cache/` (about a minute); later
loads take a second. Set the environment variable `NEUROFLY_DATA` to keep the data elsewhere.

## Check the install

```powershell
python -m pytest -q                  # about 25 seconds; includes a real-data test when the data exists
neurofly list                        # body tasks, PC tasks, connectome subsets
neurofly inspect --subset central    # what is in the brain, which populations were found
neurofly bench --subset central      # how fast it steps on this machine
```

## Five-minute tour

The brain watching a region of your screen and saying what it would press, without pressing
anything:

```powershell
neurofly play --region 0,0,800,600 --keys w,a,s,d --mouse --brain malecns --dry-run --steps 50
```

Record yourself for a minute, fit the brain to what you did, export it, and run the export
without the training stack:

```powershell
neurofly record  --window "My App" --keys w,a,s,d --mouse --out data/recordings/run1 --seconds 60
neurofly imitate data/recordings/run1 --brain malecns --run-name imitate_myapp
neurofly export  runs/imitate_myapp artifacts/myapp
neurofly-core info artifacts/myapp
neurofly-core run  artifacts/myapp --window "My App" --dry-run
```

The brain told what is on the screen by an object detector you describe in words (needs the
`detect` extra; the first run downloads the detector's weights):

```powershell
neurofly play --region 0,0,800,600 --keys w,a,s,d --brain malecns --detect "owl2:enemy,health pack" --dry-run --steps 20
neurofly detect-list      # every detector backend, its licence, and whether it is installed
```

The simulated fly, no brain, random actions, written to a video (needs the `flybody` extra):

```powershell
neurofly watch --task forward --policy random --video videos/random.mp4
```

## Where things go

| Path | Contents |
|---|---|
| `data/malecns/` | the connectome files and their cache (ignored by git) |
| `data/recordings/<name>/` | recordings of your own use of the PC, for imitation |
| `runs/<name>/` | one directory per training run: `config.json` plus weights and logs |
| `artifacts/<name>/` | exported controllers: `manifest.json` plus binary arrays |
| `videos/` | videos written by `watch` and `play --record` |

Next: [Concepts](concepts.md) for what the pieces are, or straight to [The PC](pc.md).

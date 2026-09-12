# Command reference

Two commands. `neurofly` (from the training package) builds, trains, records and exports;
`neurofly-core` (from the runtime) runs artifacts. `--help` on any subcommand prints the
current options.

## neurofly

| Subcommand | Purpose |
|---|---|
| `list` | body tasks, PC tasks and connectome subsets |
| `devices` | audio capture devices |
| `download` | fetch the connectome files |
| `inspect` | what is in the connectome and which populations were found |
| `bench` | how fast the brain steps |
| `train` | PPO |
| `es` | evolution strategies over a decoder |
| `imitate` | fit a decoder to a recording |
| `record` | record your own use of the PC |
| `play` | let the brain use the PC (during development; `neurofly-core run` after export) |
| `watch` | roll out a run and write a video |
| `export` | a run directory to an artifact directory |
| `build` | a base artifact (brain and encoders, no policy) to train from another language |
| `eval` | score an artifact offline against recordings |
| `surrogate` | train the retina's input map and a linear head through the brain |
| `calibrate` | sweep a gain for sparse, not silent, brain activity |
| `idm` | train an inverse dynamics model on labelled recordings |
| `label` | label footage that has no input log, with an inverse dynamics model |
| `replay` | a video with the brain's spikes and the controls drawn beside the frames |
| `detect-label` | run a detector over footage (`--detect`, `--every`, `--preview`) into a YOLO-layout dataset (`--out`) |
| `detect-train` | fine-tune a detector on such a dataset (`--out`, `--backend ssdlite\|rtdetr\|dfine\|yolo`, `--model`, `--epochs`, `--size`) |
| `detect-list`, `detect-install` | the detector backends with licences and install state; pip-install what some need |
| `export-body` | the fly body as a glTF (`--out`, `--task`) for Three.js and other renderers, and with `--mjcf DIR` the complete MuJoCo model; pair with `watch --poses` or `--pose-ws` |

### Environment options (train, es, imitate, play, build)

| Option | Default | Meaning |
|---|---|---|
| `--task` | `forward` | `forward`, `ball`, `pc` (the live screen), or a video file path |
| `--brain` | `none` (train), `malecns` (others) | `none`, `malecns`, `synthetic` (a random test brain) |
| `--subset` | `vnc` body / `central` PC | `vnc`, `central`, `visual`, `brain`, `no-optic`, `full` |
| `--dt` | 0.5 | brain step, ms |
| `--readout` | `motor+descending` body / `descending` PC | `motor`, `descending`, `ascending`, `cbmotor`, `visual`, joined by `+` |
| `--include-proprio` | off | body: also give the policy the raw observation |
| `--plasticity` | off | reward acts as dopamine on synapses onto the readout |
| `--brain-gain` | 1.0 | multiplier on every synapse |
| `--encoder-gain` | 12.0 | body: sensory drive, mV |
| `--synthetic-n` | 3000 | neurons in the synthetic brain |

PC options (the same commands):

| Option | Default | Meaning |
|---|---|---|
| `--fps` | 10 | steps per second for live sources |
| `--window` | | capture the window whose title contains this (Windows) |
| `--region` | | capture `left,top,width,height` |
| `--monitor` | 1 | capture a whole monitor when neither above is given |
| `--audio` | none | `loopback`, a capture device name or index, or `none` |
| `--keys` | | keys the brain may hold, comma-separated |
| `--buttons` | | mouse buttons it may hold: `left`, `right`, `middle` |
| `--mouse` | off | it may move the mouse |
| `--mouse-speed` | 50 | pixels per step at full tilt |
| `--scroll` | off | it may scroll |
| `--pad-buttons` | | gamepad buttons it may hold: `a,b,x,y,lb,rb,start,back,ls,rs,dup,ddown,dleft,dright` |
| `--axes` | | gamepad axes it may move: `lx,ly,rx,ry,lt,rt` |
| `--reward` | | `Task` giving reward and episode ends: `module:Name` or `file.py:Name` |
| `--max-steps` | | episode length cap |
| `--dry-run` | off | print the controls instead of sending them |
| `--brain-ms` | 10 | brain time per step, ms |
| `--retina-mode` | auto | `hex`, `projection`, or `auto` by subset |
| `--retina-gain` | 15 | drive at full brightness, mV |
| `--retina-temporal` | 0 | 0 = brightness, 1 = brightness change |
| `--audio-gain` | 15 | drive at full loudness, mV |
| `--include-frame` | off | also give the policy a 12 x 16 luminance grid |
| `--include-audio` | off | also give the policy the 16 audio band levels |
| `--detect` | none | object detector, `backend:arg`: `owl2:enemy,health pack`, `owl:`, `gdino:`, `yolo-world:` (open vocabulary); `yolo:`, `rtdetr:`, `dfine:` (pretrained or a `detect-train` run directory); `onnx:DIR`. `neurofly detect-list` shows them |
| `--detection-grid` | 6,8 | cells per detected class |
| `--detection-gain` | 15 | drive at full coverage of a cell, mV |
| `--include-detections` | off | also give the policy the detection grids |
| `--dopamine-punish` | 0 | mV on PPL1 dopamine neurons while reward is negative |

### train

| Option | Default | Meaning |
|---|---|---|
| `--timesteps` | 1,000,000 | total environment steps |
| `--n-envs` | 8 / 4 with brain / 1 live | parallel environments |
| `--n-steps` | 1024 | PPO rollout per environment |
| `--gamma` | 0.99 | discount |
| `--seed` | 0 | |
| `--device` | cpu | torch device for the policy and the brain |
| `--run-name` | task, brain and timestamp | directory under `runs/` |
| `--resume` | | run directory to continue |
| `--checkpoint-every` | 200,000 | steps between checkpoints |

### es

| Option | Default | Meaning |
|---|---|---|
| `--generations` | 100 | |
| `--population` | 32 | candidates per generation, even |
| `--sigma` | 0.05 | noise scale |
| `--lr` | 0.03 | step size |
| `--episode-steps` | 500 | steps per rollout |
| `--workers` | half the CPUs / 1 live | parallel processes |
| `--seed`, `--run-name` | | |

### imitate

Positional: one or more recording directories.

| Option | Default | Meaning |
|---|---|---|
| `--holdout` | 0.2 | fraction of each recording kept for evaluation |
| `--max-frames` | | cap per recording |
| `--epochs` | 300 | |
| `--lr` | 0.01 | |
| `--l2` | 1e-4 | weight decay |
| `--seed`, `--device`, `--run-name` | | |

`--audio` defaults to the recording's own sound; `--audio none` ignores it.

### record

| Option | Default | Meaning |
|---|---|---|
| `--out` | required | output directory |
| `--keys`, `--buttons`, `--mouse`, `--mouse-speed`, `--scroll` | | what to log (the layout) |
| `--window`, `--region`, `--monitor` | | what to capture |
| `--audio` | none | `loopback`, a device, or none |
| `--fps` | 10 | |
| `--size` | 320,240 | stored frame size |
| `--seconds` | | stop after this long |
| `--countdown` | 3 | seconds before starting |
| `--panic` | esc | key that stops the recording |

### play

| Option | Default | Meaning |
|---|---|---|
| `--run` | | run directory with a decoder or PPO model; its settings are loaded |
| `--policy` | the run's, else `fixed` | `ppo`, `es`, `imitation`, `fixed` (random table), `random` |
| `--steps` | | stop after this many steps |
| `--countdown` | 3 | seconds to focus the target window |
| `--panic` | esc | key that stops everything |
| `--record` | | save captured frames to this mp4 |
| `--seed`, `--device` | | |

### watch

Positional: a run directory (optional).

| Option | Default | Meaning |
|---|---|---|
| `--policy` | the run's, else `zero` | `ppo`, `es`, `imitation`, `fixed`, `random`, `zero` |
| `--task`, `--brain`, `--subset` | from the run | overrides |
| `--keys`, `--buttons`, `--mouse`, `--window`, `--region` | from the run | PC overrides |
| `--episodes` | 2 | |
| `--max-steps` | | cap per episode |
| `--seed` | 123 | |
| `--stochastic` | off | sample the PPO policy instead of its mean |
| `--video` | | write an mp4 |
| `--camera` | 1 | body: MuJoCo camera |
| `--every` | 10 body / 1 PC | control steps per video frame |
| `--poses` | | body: write every body's world pose per rendered frame to this JSON |
| `--pose-ws HOST:PORT` | | body: stream poses live over a WebSocket, pacing the simulation to real time |

On the live screen `watch` is always a dry run.

### export

Positional: the run directory and the artifact directory to write. `--name` sets the
artifact's name (default: the run's directory name); `--device` picks where the model is
rebuilt. The artifact is validated after writing.

### build

Positional: the artifact directory to write. Takes the environment and PC options above
(`--brain`, `--subset`, `--keys`, `--buttons`, `--mouse`, `--audio`, `--brain-ms`,
`--include-frame`, ...) plus `--name` and `--device`. The artifact is validated after
writing; it has no policy, so `neurofly-core run` refuses it until one is installed through
the server's `set_policy` and `save` (see [Runtime](runtime.md)).

### eval

Positional: the artifact, then recording directories. `--reward module:Name` scores reward
under a Task; `--max-frames` caps each recording; `--out` names the JSON report (default
`<artifact>/eval.json`); `--device`.

### surrogate

Positional: recording directories. Takes the environment options (`--brain`, `--subset`,
`--brain-ms`, `--audio`, ...) plus `--epochs` (5), `--lr` (0.01), `--bptt-window` (frames per
gradient step, 8), `--l2`, `--holdout`, `--max-frames`, `--seed`, `--device`, `--run-name`.
Writes `runs/<name>/artifact`.

### calibrate

Takes the environment options plus `--video FILE` (frames from a video instead of the
screen), `--frames` (per gain value, 8), `--param` (`brain_gain`, `retina_gain`,
`audio_gain`), `--grid` (comma-separated values to try), `--target` (wanted fraction of
active readout neurons, 0.2). Prints a table and the pick.

### idm

Positional: labelled recording directories. `--context` (frames before and after, 2),
`--grid` (luminance grid, `12,16`), `--hidden` (256), `--epochs` (200), `--lr`,
`--holdout`, `--max-frames`, `--seed`, `--run-name`. Writes `runs/<name>/idm.pt` and
`idm.json`.

### label

Positional: video files. `--idm RUN` (from `idm`), `--out DIR` (one recording directory per
video is written under it), `--max-frames`.

### replay

`--video FILE` or `--recording DIR` (frames; a recording also supplies the controls),
`--probe FILE` (from `--probe-out`), `--out FILE`, `--history` (steps of raster shown, 50),
`--panel` (side panel width, 320), `--max-frames`.

### play: corrections and safety

`--correct-out DIR` records your inputs as labels whenever you touch a control in the
layout; the fly stands down while you do (see [The PC](pc.md)). `--no-focus-guard` keeps
the fly going when the keyboard focus leaves the window it started on; by default it stops.
`neurofly-core run` has the same `--no-focus-guard`.

### Experiments (play, watch, neurofly-core run and serve)

| Option | Meaning |
|---|---|
| `--stimulate SEL:MV` | extra drive on a selection every brain step; repeatable |
| `--silence SEL` | the selection never spikes; repeatable |
| `--probe SEL` | record the selection's spikes and rates every step |
| `--probe-out FILE` | write the probe as `.npz` at the end |
| `--activity-out FILE.json` | record every neuron's spikes per step, with positions, for `examples/brain_viewer.html` |
| `--activity-ws HOST:PORT` | stream the same live over a WebSocket (`brain_viewer.html?ws=...`) |
| `--activity-substeps` | activity per brain step rather than per observation |

`SEL` is `indices=1,2`, `ids=...`, `type_re=^PPL1`, `superclass=descending_neuron`, or
`name=readout|retina|audition|punish`.

### bench

| Option | Default | Meaning |
|---|---|---|
| `--brain` | malecns | or `synthetic` |
| `--subset` | vnc | |
| `--dt` | 0.5 | |
| `--device` | cpu | |
| `--backend` | auto | `event`, `torch` |
| `--steps` | 200 | |
| `--drive-mv` | 12 | constant drive on the leg proprioceptive neurons (or the visual projection neurons) |

### inspect

`--subset` (default `full`), `--synthetic`, `--data-dir`.

### download

`--data-dir`, `--overwrite`.

## neurofly-core

| Subcommand | Purpose |
|---|---|
| `info <artifact>` | describe an artifact |
| `validate <artifact>` | check files, sizes, shapes and index ranges; exit 1 on problems |
| `serve <artifact>` | JSON lines over stdio; `--ws HOST:PORT` for a WebSocket; `--grpc HOST:PORT` for gRPC; `--device`; the experiment options |
| `run <artifact>` | capture the screen and drive keyboard, mouse and gamepad; the experiment options |

### run

| Option | Default | Meaning |
|---|---|---|
| `--window`, `--region`, `--monitor` | | what to capture |
| `--audio` | none | `loopback`, a device, or none |
| `--fps` | 10 | |
| `--steps` | | stop after this many steps |
| `--countdown` | 3 | seconds to focus the target window |
| `--panic` | esc | key that stops everything |
| `--dry-run` | off | print the controls instead |
| `--device` | cpu | |

# The PC

The PC is the fly's second world: what is on the screen and what the PC is playing go in,
keyboard and mouse come out. Everything is in `neurofly_core.io` (sources and controls) and `neurofly_training.pc` (the environment, tasks, imitation) and every piece is a base
class with a few implementations, meant to be swapped or subclassed.

## The pieces

| Piece | Base class | Implementations |
|---|---|---|
| video in | `VideoSource` | `ScreenCapture` (a window by title, a region, or a monitor), `VideoFile`, `SyntheticVideo` |
| sound in | `AudioSource` | `AudioCapture` (a capture device, or `loopback` = what the PC plays), `AudioFile`, `SilentAudio`, `SyntheticAudio` |
| controls out | `Controls` | `PCControls` (the real keyboard and mouse), `LoggingControls` (prints instead), `NullControls` |
| what it may touch | `ControlLayout` | keys, mouse buttons, mouse motion, scroll |
| reward and episodes | `Task` | `NoTask`, `PatchBrightness`; yours |
| the loop | `PCEnv` | a Gymnasium environment |

### The control layout and the action vector

A `ControlLayout` lists the keys and mouse buttons the brain may hold and whether it may
move the mouse or scroll. It packs them into one action vector in [-1, 1]:

| Entries | Meaning |
|---|---|
| one per key | the key is held while the entry is positive |
| one per button | the button is held while positive |
| two (if `--mouse`) | mouse motion this step, as a fraction of `--mouse-speed` pixels (default 50) |
| one (if `--scroll`) | wheel clicks this step, as a fraction of 3 |

Key names are single characters (`a`, `1`) or these words: `ctrl`, `shift`, `alt`, `space`,
`enter`, `esc`, `tab`, `backspace`, `up`, `down`, `left`, `right`, `f1` ... `f12`. Buttons
are `left`, `right`, `middle`. On the command line: `--keys w,a,s,d,space --buttons left
--mouse --scroll`.

A layout can also hold a gamepad: `--pad-buttons a,b,x,y,lb,rb,start,back,ls,rs,dup,ddown,
dleft,dright` (one entry each, held while positive) and `--axes lx,ly,rx,ry,lt,rt` (sticks
in [-1, 1], triggers in [0, 1]). `record` reads a real controller through XInput; `play`
and `neurofly-core run` drive a virtual Xbox controller through ViGEm (`pip install
vgamepad` plus the ViGEmBus driver, Windows only). Keys, mouse and pad can be mixed.

### One step

1. The action vector is unpacked into a `ControlState` and applied: keys and buttons that
   changed are pressed or released, the mouse moves, the wheel scrolls.
2. With a live source, the loop waits for the next tick (`--fps`, default 10 per second).
3. The new frame is read; with sound, everything captured since the last step is read.
4. The `Task` computes the reward and decides whether the episode is over.
5. The frame drives the retina and the sound drives the auditory neurons; the brain runs
   `--brain-ms` milliseconds (default 10, that is 20 sub-steps at `--dt 0.5`).
6. The agent sees the readout rates (by default the descending neurons), plus a 12 x 16
   luminance grid with `--include-frame` and the 16 audio band levels with `--include-audio`.

Episodes also end at `--max-steps`, and, for a video file, at the end of the file.

## Watch what it would do

Nothing is sent to the PC with `--dry-run`; the controls are printed instead. Start there:

```powershell
neurofly play --window "My App" --keys w,a,s,d --mouse --brain malecns --dry-run
neurofly play --region 0,0,800,600 --keys w,a,s,d --brain malecns --dry-run --steps 100
```

`--window` captures the client area of the first visible window whose title contains the
text (Windows only). `--region left,top,width,height` captures a screen rectangle anywhere.
`--monitor N` captures a whole monitor. Frames are captured at 320 x 240.

## Let it use the PC

Without `--dry-run`, key presses and mouse motion go wherever the keyboard focus is. Focus the
target window during the countdown. **Esc stops everything** and releases every key and
button (`--panic` chooses another key). `--steps` stops after that many steps.

```powershell
neurofly play --window "My App" --keys w,a,s,d --buttons left --mouse --brain malecns --policy fixed --seed 3
neurofly play --window "My App" --run runs/<a PPO, ES or imitation run>
neurofly play --window "My App" --run runs/<run> --audio loopback --record videos/live.mp4
```

`--policy fixed` is a random neuron-to-control table (the seed picks which). `--run` loads a
trained one, along with the layout and brain settings it was trained with; options you pass
explicitly override those. `--record` saves the captured frames to a video.

## Sound

`--audio loopback` captures what the PC is playing on its default output. `--audio <name or
index>` captures a device (a microphone, a line input); `neurofly devices` lists
them. Sound goes to the auditory neurons through the audition encoder, and `--include-audio`
also gives the policy the band levels directly.

## Objects on screen: detectors

A fly cannot learn from pixels what an enemy is. An object detector can say where the
enemies are, and a *detection encoder* turns that into drive on central-brain neurons, one
labelled line per class and place. `--detect` names the detector on `train`, `es`,
`imitate`, `surrogate`, `play`, `watch`, `build` and `eval`; the classes go into the
artifact so a consumer in any language knows which ids to send.

```powershell
pip install -e training[detect]
neurofly play --window "My App" --keys w,a,s,d --detect "owl2:enemy soldier,health pack,door" --dry-run
```

A detector is a *backend* behind one spec grammar, `backend:arg`. `neurofly detect-list`
shows them all, whether they are installed, and their licences; `neurofly detect-install
<backend>` pip-installs what one needs into the running Python.

| Spec | What | Licence | On a CPU |
|---|---|---|---|
| `owl2:enemy,health pack` | OWLv2, open vocabulary: name the objects, no training | Apache-2.0 | 3 s per frame |
| `owl:...` | OWL-ViT: the same, three times faster, weaker on screens | Apache-2.0 | 1 s |
| `gdino:...` | Grounding DINO: phrase-grounded boxes | Apache-2.0 | several seconds |
| `yolo-world:enemy,door` | YOLO-World v2: open vocabulary in real time | AGPL package, GPL weights | 50 ms |
| `yolo:` / `yolo:yolo11m.pt` | YOLO11 on its 80 COCO classes; also the best fine-tuning tooling | AGPL-3.0 | 20 to 60 ms |
| `rtdetr:` | RT-DETRv2, pretrained on COCO; fine-tunable | Apache-2.0 | 100 to 200 ms |
| `dfine:` | D-FINE, the current accuracy leader per FLOP; fine-tunable | Apache-2.0 | 100 to 200 ms |
| `runs/det1` | any `detect-train` run directory | as its backend | |
| `onnx:runs/det1` | an exported `detector.onnx` through ONNX Runtime, the same file a consumer in another language runs | the model's own | |

Open-vocabulary detectors need nothing from you. Check what a prompt finds before trusting
it: `neurofly detect-label footage.mp4 --detect "owl2:enemy" --preview check.mp4`. Scores
are low by nature; 0.1 is a sensible cut for OWL, 0.25 for the others.

When you want speed or accuracy, distil the open-vocabulary detector into a fast one, still
without labelling anything by hand:

```powershell
neurofly detect-label footage/*.mp4 --detect "owl2:enemy soldier,health pack" --out data/objects --every 5
neurofly detect-train data/objects --out runs/det1 --backend dfine --epochs 30
neurofly play --window "My App" --keys w,a,s,d --detect runs/det1
```

`detect-label` runs the detector over footage and writes a dataset in the YOLO layout
(`images/`, `labels/`, `classes.json`, `data.yaml`). If some boxes are wrong, any
YOLO-format labelling tool opens that folder and you fix only what matters. `detect-train`
fine-tunes the backend you choose: `ssdlite` (the small fast baseline, exports
`detector.onnx`), `rtdetr` or `dfine` (strong, Apache) or `yolo` (Ultralytics, AGPL).
The run directory is then the spec.

Detections on a recording are computed once and cached next to the video. `--detection-grid`
sets the cells per class, `--detection-gain` the drive, and `--include-detections` also
hands the grids to the policy directly.

## Smell: odours the brain can learn about

The fly smells through some fifty kinds of receptor neuron, one kind per glomerulus, and
its learning circuit (the mushroom body, taught by the dopamine neurons that
`--dopamine-punish` already drives) is built around them. `--odours` gives the brain that
input: each named channel drives the receptor neurons of one glomerulus, so "danger" or
"low health" is a smell the brain's own associative memory can attach the punishment to.

```powershell
neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d \
    --reward my_project.py:MyTask --odours health,danger --plasticity --dopamine-punish 20
```

Where the values come from:

* **Your Task.** Add an `odours(frame, audio, info)` method returning `{"health": 0.8,
  "danger": 0.0}` (values in [0, 1]; a missing channel keeps its last value) or a vector
  in channel order. Health bars, ammo, distance to the goal, "in combat": anything slow
  that the fly should learn to like or avoid.
* **Detections.** `--odours detections` makes one channel per detected class, set to the
  best score of that class in the frame, so an enemy in view also has a smell.
* **Another program.** `odours` on the step request, in the JSON protocol, gRPC and the
  bindings; the artifact lists the channels it expects.

`--odour-gain` is the drive at a channel value of 1; `--odour-adapt` lets the drive fade
while a channel stays constant, as receptors do; `--include-odours` also hands the channel
values to the policy directly. The channels travel in the artifact; the meaning of each is
yours.

## Reward: writing a Task

The raw interface only sees pixels and sound. A `Task` turns them into reward and episode
structure. `PatchBrightness` is the built-in example: reward is the mean brightness of a
rectangle of the frame, given as fractions of the frame size, which is enough to read a
health bar or a progress bar:

```powershell
neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d \
    --reward neurofly_training.pc.task:PatchBrightness --max-steps 600
```

Your own is a subclass in any file, referenced as `file.py:Name` or `package.module:Name`:

```python
# my_project.py
import numpy as np
from neurofly_training.pc.task import Task

class Score(Task):
    def reset(self, controls, video, audio):
        # start an episode: e.g. press the key that restarts the thing
        from neurofly_core.controls import ControlState
        controls.apply(ControlState(keys=frozenset({"r"})))
        controls.release_all()

    def reward(self, frame, audio, state, info):
        bar = frame[10:14, 20:220]                  # a strip of the screen
        info["health"] = float((bar[..., 1] > 128).mean())
        return info["health"] - 0.5

    def odours(self, frame, audio, info):
        # with --odours health: a smell the mushroom body can learn about
        return {"health": info.get("health", 0.0)}

    def done(self, frame, audio, info):
        return info["health"] == 0.0
```

```powershell
neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d --mouse \
    --reward my_project.py:Score --max-steps 600 --dopamine-punish 20
```

`reward` receives the frame (RGB uint8), the audio chunk since the last step (or `None`),
the `ControlState` just applied, and the step's `info` dict (which you may add to; `watch`
and `play` print nothing from it, but PPO's monitor and your own code can read it). `reset`
receives the `Controls`, so it can press keys, and the sources. `done` ends the episode.

Live training uses one environment (there is one screen), so `train` forces `--n-envs 1`
and `es` forces `--workers 1`.

## Safety

Without `--dry-run`, the fly sends real input. Two guards run by default on `play` and
`neurofly-core run`: a **focus guard** that stops the loop, and releases everything, the
moment the keyboard focus leaves the window it started on (so the fly cannot type into
whatever you switched to; `--no-focus-guard` disables it), and a **watchdog** thread that
releases every key and button if the loop stalls for more than a second. Esc remains the
panic key.

## Reading the screen for reward

Two helpers in `neurofly_training.pc.screen` cover most rewards on real programs:

* `TemplatePresence(template.png, region, end_below)`: reward is how well a small image
  (an icon, a marker) matches inside a region of the frame, by normalised cross-correlation;
  the episode can end when it disappears.
* `NumberOnScreen(region, scale)`: reward is the change of a number on screen (a score, a
  counter) between steps, read by OCR. Needs `pip install pytesseract` and a Tesseract
  install; `read_number(frame, region)` is the underlying call.

Both are `Task` subclasses, so `--reward neurofly_training.pc.screen:TemplatePresence`
works once you subclass them with your region and template, and both compose with your
own `Task` code.

## Correcting the fly while it plays

`neurofly play --run runs/<run> --correct-out data/recordings/corrections` watches your
keyboard, mouse and controller while the fly plays (ignoring the fly's own injected input).
The moment you touch a control in the layout, the fly's controls are released and the
frame plus your action are recorded; when you let go, the fly resumes. Train on the
correction recording together with the originals afterwards:

```powershell
neurofly imitate data/recordings/run1 data/recordings/corrections --run-name corrected
```

That is the DAgger loop: the policy learns what you did in the states it actually reaches,
which is where pure imitation goes wrong.

## Recording yourself and imitating

`record` captures the screen, optionally the sound, and what you hold and move, at `--fps`:

```powershell
neurofly record --window "My App" --keys w,a,s,d --buttons left --mouse --audio loopback --out data/recordings/run1
```

Esc stops it (or `--seconds`). A recording directory holds `video.mp4`, `audio.wav` (with
`--audio`), `actions.npy` (one action vector per frame, in the layout you gave) and
`meta.json` (the layout, fps, size, sample rate). Mouse motion is measured from the cursor
position, so a program that hides and re-centres the cursor will not report it.

`imitate` plays the recordings through the brain, collects the readout on every frame, and
fits a linear map from readout to what you did:

```powershell
neurofly imitate data/recordings/run1 data/recordings/run2 --brain malecns --run-name imitate_myapp
neurofly watch runs/imitate_myapp --task data/recordings/run1/video.mp4 --video videos/imitate.mp4
neurofly play  --window "My App" --run runs/imitate_myapp
```

It reports, on the last `--holdout` fraction of each recording, precision, recall and F1 per
key and button, and correlation and mean absolute error for mouse and scroll. A control that
scores near its base rate means the brain's response to the input does not carry that
information. The knobs: more brain (`--subset visual`), more brain time (`--brain-ms 20`),
a different readout (`--readout descending+cbmotor+visual`), or the raw grid and bands
(`--include-frame --include-audio`).

The recordings' own sound is used when present; `--audio none` ignores it.

## Offline: a video file as the world

Any video file is a task: `--task path/to/video.mp4`. Frames are read in order, nothing is
sent to the PC, and the episode ends with the file. This is what `imitate` uses, and it is
useful for testing a `Task`'s reward on footage before going live:

```powershell
neurofly watch --task data/recordings/run1/video.mp4 --brain malecns --keys w,a,s,d --policy fixed --max-steps 100
neurofly es --task data/recordings/run1/video.mp4 --brain malecns --keys w,a,s,d --reward my_project.py:Score --episode-steps 100
```

## Limits worth knowing

* Window lookup by title is Windows-only; use `--region` elsewhere.
* Keys are sent as system input events. Programs that read the keyboard at a lower level, or
  that reject synthetic input, will not see them.
* Mouse motion is relative (`move dx, dy`). Programs that warp the cursor to the centre every
  frame still receive the motion, but recordings of you in such programs will not capture yours.
* Screen capture reads what is composited on screen: a minimised window has no frames.
* The brain does not stop when the script crashes, but every exit path releases all keys
  and buttons; if something is stuck, tap the keys yourself.

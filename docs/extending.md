# Extending

Everything is a plain Python class with a small interface. This page shows the seams. Two
packages: `neurofly_core` (runtime: brain, encoders, decoders, PC input and output, model,
artifact, server) and `neurofly_training` (connectome, building, environments, training,
export).

## The Python API in one example

```python
import numpy as np
from neurofly_core.controls import ControlLayout
from neurofly_core.io.audio import AudioCapture
from neurofly_core.io.controls import LoggingControls, PCControls
from neurofly_core.io.video import ScreenCapture
from neurofly_training.build import build_model
from neurofly_training.envs import load_connectome
from neurofly_training.pc.env import PCEnv
from neurofly_training.pc.task import PatchBrightness

layout = ControlLayout(keys=["w", "a", "s", "d"], buttons=["left"], mouse=True)
cx = load_connectome("malecns", subset="central")
model = build_model(cx, layout, readout="descending", brain_ms=10, audio=True)   # a neurofly_core.Model
env = PCEnv(model,
            video=ScreenCapture(window="My App", fps=10),
            audio=AudioCapture(loopback=True),
            controls=LoggingControls(),                 # PCControls() to really press things
            task=PatchBrightness(patch=(0.0, 0.9, 1.0, 0.1)),
            fps=10, max_steps=300)
obs, _ = env.reset()
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(info["held"], info["brain_spikes"], reward)
env.close()

model.policy = ...                                       # a ControlDecoder or MLPPolicy
from neurofly_core.artifact import save_model
save_model(model, "artifacts/myapp")                     # runs anywhere from here
```

`make_env(...)` in `neurofly_training.envs` builds the same thing from keyword arguments
matching the command-line options; `make_pc_env(...)` accepts ready-made objects
(`video_source`, `audio_source`, `controls`, `task_obj`, `model`, `layout`) alongside them;
`make_pc_model(...)` builds only the model, which is what `export` uses.

The body side is the same shape: `make_env("forward", brain="malecns")` returns a
`BrainInLoopEnv` whose `.brain`, `.pops`, `.encoder` and `.readout_idx` are yours to inspect.

## The Model

`neurofly_core.Model` is the runtime object: a `LIFBrain`, a `RetinaEncoder`, optionally an
`AuditionEncoder`, the readout indices, a `ControlLayout`, optionally a policy, and a
`ModelConfig` (brain time per observation, warm-up, feature options, plasticity). Three
methods: `reset()`, `observe(frame, audio, reward) -> features`, `act(features) -> action`;
`step` is both. `PCEnv` calls `observe` and supplies its own actions during training; a
deployed controller calls `step`. `save_model` / `load_model` round-trip it through the
artifact format.

## A Task

```python
from neurofly_training.pc.task import Task

class MyTask(Task):
    def reset(self, controls, video, audio):      # optional: start an episode
        ...
    def reward(self, frame, audio, state, info):  # frame: RGB uint8; audio: (n, ch) float or None
        return 0.0
    def done(self, frame, audio, info):
        return False
```

Point the commands at it with `--reward my_project.py:MyTask` or
`--reward my_package.module:MyTask`. The loader instantiates a class with no arguments,
accepts an instance, or calls a factory function. `state` is the `ControlState` just applied
(`.keys`, `.buttons`, `.dx`, `.dy`, `.scroll`); `info` is the step's dict, shared with the
caller. `examples/health_bar_task.py` is a complete one.

## A video source

```python
from neurofly_core.io.video import VideoSource

class MySource(VideoSource):
    live = True          # True: the env paces itself to fps and never runs out of frames
    fps = 30.0

    def read(self):      # RGB uint8 (h, w, 3), or None when a file source is finished
        ...

    def close(self):
        ...
```

Use it with `PCEnv(model, video=MySource())` or `make_pc_env(..., video_source=MySource())`.
The retina reduces any frame size to its 24 x 32 grid, so resolution is your choice.

## An audio source

```python
from neurofly_core.io.audio import AudioSource

class MySound(AudioSource):
    live = False
    sample_rate = 16000
    channels = 1

    def read(self):      # float32 (n, channels) in [-1, 1]: since the last read, or the next 1/fps s
        ...
```

## Controls

Subclass `Controls` and override the six hooks; the base class tracks what is held and calls
them only on changes:

```python
from neurofly_core.io.controls import Controls

class SerialControls(Controls):
    def _press_key(self, name): ...
    def _release_key(self, name): ...
    def _press_button(self, name): ...
    def _release_button(self, name): ...
    def _move(self, dx, dy): ...
    def _scroll(self, n): ...
```

This is the seam for driving something other than the local keyboard and mouse: a
network-connected machine, a virtual gamepad, a robot. Outside Python, the same role is
played by whatever applies the server's responses.

## A new encoder

At runtime an encoder is tables plus a rule: anything with `reset()`,
`__call__(input) -> torch.Tensor` of shape `(n_neurons,)` in mV, and, to travel in an
artifact, `params()` / `tables()` / `from_tables()`. `neurofly_training.build` is where
tables are made from the connectome's annotations: `Populations` gives you the neuron
indices to target, and `Connectome.select(...)` any other selection by annotation column,
for example `cx.select(superclass="cb_sensory", subclass="grooming")`. Add a builder there,
give the `Model` a new attribute, and teach `artifact.py` to save and load it.

## A new readout

`Populations.readout(name)` resolves `+`-joined names from a dict of populations. Add an
entry to `Populations._readouts`, or pass an index array instead of a name:
`build_model(cx, layout, readout=cx.select(type_re="^DNa"))`.

## A new connectome subset

`SUBSETS` in `neurofly_training.data.connectome` maps a name to a function returning the
neuron indices to keep. Anything you can express with the annotation columns works, and
`Connectome.neighborhood(seeds, hops)` grows a set along the synapses.

## A new body task

Subclass `flybody.tasks.base.Walking`, write `get_reward_factors` and `check_termination`,
wrap it in a factory returning a `composer.Environment`, and register the factory in
`TASKS` in `neurofly_training.body.tasks`. `forward` in the same file is the template.

## A new training method

An environment is a standard Gymnasium env: `BrainInLoopEnv` and `PCEnv` both pass
Stable-Baselines3's `check_env`. Any algorithm that takes a Gymnasium env works. Decoders
expose `get_params()` / `set_params()` on a flat vector, which is all evolution strategies
or any other black-box optimiser needs. To export a new kind of policy, give it `params()`
and `tables()` (see `neurofly_core.decode.mlp`) and a branch in `artifact.load_model`.

## A detector backend

Detectors are entries in `neurofly_training.pc.backends.BACKENDS`: a name (the spec
prefix), the pip packages it needs, its licence, a default model, and dotted paths to a
factory `(arg, device, threshold) -> Detector` and, for trainable ones, a trainer
`(dataset, out, **options) -> history`. A `Detector` has `classes` (names in id order),
`detect(frame) -> [{"class", "box", "score"}]` with the box in fractions of the frame, and
optional `reset()` / `close()`. Add the entry and the factory (the existing ones in
`neurofly_training.pc.detect` are ten to forty lines each) and `--detect name:arg`,
`detect-list`, `detect-install` and `detect-train --backend name` all know about it.

## A new consumer language

Nothing to add to this repository: read [artifact/SPEC.md](../artifact/SPEC.md) and either
spawn `neurofly-core serve` and speak JSON lines (the Node and Rust bindings are under 200
lines each), or read the arrays and implement the dynamics natively.

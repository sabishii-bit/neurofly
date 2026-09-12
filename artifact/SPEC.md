# The neurofly artifact

A trained controller on disk, readable from any language. Version 1.

## Layout

```
<artifact>/
  manifest.json          everything that is not an array
  brain/                 indptr.bin  indices.bin  values.bin
  readout/               indices.bin
  retina/                indices.bin pixels.bin on.bin          (hex mode)
                         matrix_*.bin targets.bin              (projection mode)
  audition/              matrix_*.bin targets.bin              (optional)
  detection/             matrix_*.bin targets.bin              (optional)
  policy/                W.bin b.bin  |  W0.bin b0.bin ... obs_mean.bin obs_var.bin   (optional)
  punish/                indices.bin                           (optional)
  neurons/               ids.bin annotations.json              (optional, provenance)
                         positions.bin positions_known.bin     (optional, a map of the brain)
```

Every array is a raw little-endian binary with no header. The manifest describes each one:

```json
{"file": "brain/values.bin", "dtype": "float32", "shape": [9464056]}
```

`dtype` is one of `float32`, `float64`, `int64`, `int32`, `bool` (one byte per value).
Multi-dimensional arrays are row-major (C order).

## Two kinds

`kind` in the manifest is `"pc"` (frames and sound in, keyboard, mouse and gamepad out;
the default when absent) or `"body"` (a body observation vector in, actuator commands out).
Both share the brain, the readout, the policy, the punishment neurons and the neuron
annotations; they differ in the encoders and the output layout.

## manifest.json

| Key | Meaning |
|---|---|
| `format` | `"neurofly-artifact"` |
| `version` | `1` |
| `kind` | `"pc"` or `"body"` |
| `name`, `created` | a name and an ISO timestamp |
| `config` | the model's loop settings (below) |
| `brain` | the neurons and synapses and the LIF parameters (below) |
| `readout.indices` | int64 array: the neurons whose firing rates are the features |
| `retina` | `params` and `tables` of the frame encoder (below) |
| `audition` | `params` and `tables` of the sound encoder, or `null` |
| `detection` | `params` and `tables` of the detected-object encoder, or `null` (below) |
| `layout` | which controls the action vector holds (below) |
| `features.n`, `actions.n` | feature and action vector lengths |
| `policy` | `params` and `tables` of the policy, or `null` (then only `observe` is possible) |
| `punish.indices` | int64 array: neurons driven while reward is negative, or `null` |
| `neurons.ids` | int64 array: the source connectome's id of every neuron, or `null` |
| `neurons.positions` | float32 array `[n_neurons, 3]`: soma position in micrometres (optional, see below) |
| `extra` | free-form provenance (the training run's config) |

### config

| Key | Meaning |
|---|---|
| `dt` | brain step in ms |
| `brain_ms` | brain time per observation; `brain_ms / dt` steps per frame |
| `warmup_ms` | extra brain time on the first frame after a reset |
| `readout_scale` | features = firing rate in Hz times this (0.01) |
| `include_frame`, `frame_grid` | append a `rows x cols` luminance grid of the frame to the features |
| `include_audio` | append the audio band levels to the features |
| `include_detections` | append the detection grids to the features |
| `plasticity`, `plasticity_lr`, `dopamine_punish` | optional online learning; a consumer may ignore them |

### brain

`n_neurons`, `n_synapses`, and the LIF parameters `dt`, `tau_m`, `tau_syn`, `v_rest`,
`v_reset`, `v_th`, `t_ref`, `rate_tau` (all ms or mV). `weights_layout` is `csc` or `csr`:

* `csc`: `indptr` has `n_neurons + 1` entries; the synapses of presynaptic neuron `j` are
  `k in [indptr[j], indptr[j+1])`, targeting postsynaptic neuron `indices[k]` with weight
  `values[k]` in mV per spike (sign included). This is the layout for event-driven
  propagation: for every neuron that spiked, add its column to the synaptic input.
* `csr`: the same by postsynaptic neuron (row `i` lists its inputs).

The dynamics, per step of `dt` ms, for every neuron:

```
I_syn  = I_syn * (1 - dt / tau_syn) + sum of weights of presynaptic spikes at the previous step
v      = v + (v_rest - v + I_syn + I_ext) * dt / tau_m
v      = v_reset  where refractory > 0
refractory = max(refractory - dt, 0)
spike  = v >= v_th ; then v = v_reset, refractory = t_ref
rate   = rate * (1 - dt / rate_tau) + spike * 1000 / rate_tau          (Hz)
```

`I_ext` is the sum of the encoders' drive vectors (mV). All state starts at `v_rest` / 0.

### retina

`params.mode` is `hex` or `projection`; `params.grid` is `[rows, cols]`; `params.gain` is mV
at full brightness; `params.temporal` in [0, 1] mixes brightness with brightness change.

The frame is converted to grayscale in [0, 1] and area-averaged to the grid (row-major,
`pixel = row * cols + col`). With `prev` the previous grid (or the same grid on the first
frame) and `t = temporal`:

```
on  = (1 - t) * lum + t * clip(4 * max(lum - prev, 0), 0, 1)
off = (1 - t) * (1 - lum) + t * clip(4 * max(prev - lum, 0), 0, 1)
```

* `hex`: for each `i`, neuron `indices[i]` gets `gain * (on if on[i] else off)[pixels[i]]`.
* `projection`: `y = matrix @ on` (CSR `matrix_indptr`, `matrix_indices`, `matrix_values`,
  shape `n_neurons x rows*cols`); over the `targets`, `z = (y - mean) / (std + 1e-6)`;
  each target gets `gain * max(z + 0.5, 0)`.

### audition

`params`: `sample_rate`, `n_bands`, `fmin`, `fmax`, `gain`, `loud_ref`. Band edges are
`n_bands + 1` values geometrically spaced from `fmin` to `fmax`. For the samples since the
last step (mono, float in [-1, 1]), with fewer than 32 samples reuse the last band levels:

```
rms = sqrt(mean(x^2)); if rms < 1e-5: bands = 0
spec = |rfft(x * hann)|^2 ; e[b] = mean of spec over frequencies in [edge_b, edge_b+1)
lvl = log10(e + 1e-12) ; shape = (lvl - min) / (max - min + 1e-6)
bands = shape * min(1, rms / loud_ref)
drive = gain * max(matrix @ bands, 0)          (CSR, n_neurons x n_bands)
```

### detection

`params`: `classes` (names, in id order), `grid` `[rows, cols]`, `gain`. The consumer runs a
detector of its own choosing (the training package ships several; `config.meta.detect`
records which one the artifact was built with) and sends, per step, boxes with class ids in
this list, as fractions of the frame. For each class `c` a `rows x cols` grid holds, per
cell, the largest score-weighted fraction of the cell covered by a box of class `c`; the
grids are flattened class-major, then row, then column, into a vector `g` of length
`n_classes * rows * cols`; `drive = gain * max(matrix @ g, 0)` (CSR `matrix_indptr`,
`matrix_indices`, `matrix_values`, shape `n_neurons x len(g)`), on the `targets`. An artifact
without `detection` ignores detections.

### features

`[rate[readout[i]] * readout_scale for i] ++ (luminance grid if include_frame) ++ (bands if include_audio) ++ (detection grids if include_detections)`

### layout

`keys` (list of names), `buttons` (`left`, `right`, `middle`), `pad_buttons` (`a`, `b`,
`x`, `y`, `lb`, `rb`, `start`, `back`, `ls`, `rs`, `dup`, `ddown`, `dleft`, `dright`),
`mouse` (bool), `scroll` (bool), `axes` (`lx`, `ly`, `rx`, `ry`, `lt`, `rt`), `mouse_speed`
(pixels per step at full deflection), `scroll_speed` (clicks). The action vector is

    [keys..., buttons..., pad_buttons..., dx, dy (if mouse), scroll (if scroll), axes...]

in [-1, 1]: a key or button is held while its entry is positive; `dx`, `dy` are multiplied
by `mouse_speed`, scroll by `scroll_speed`; stick axes are the entry itself; the triggers
`lt` and `rt` map `[-1, 1]` to `[0, 1]`.

### neurons

`neurons.ids` (int64) is the source connectome's id of every neuron, and
`neurons.annotations` names a JSON file `{"type": [...], "superclass": [...]}` with one
string per neuron (empty when unknown). They make selections by type or superclass possible
at runtime (below); a consumer that does not need them can ignore them.

`neurons.positions` (float32, `[n_neurons, 3]`, `neurons.positions_unit` = `"micrometre"`)
is where each neuron's soma is in the source volume, for drawing the brain. Neurons with no
soma in the volume (sensory neurons) are given a position near their annotated group;
`neurons.positions_known` (bool, `[n_neurons]`) is true for real somas and false for placed
ones. The frame is the connectome's own (MaleCNS: x right, y down, z front to back); it is
not related to any body model.

### body (kind body only)

| Key | Meaning |
|---|---|
| `actuators` | the actuator names in order; the action vector has one entry each, in [-1, 1] |
| `obs_dim`, `obs_keys`, `obs_slices` | the observation vector: its length, the observables in order, and `{name: [start, stop]}` |
| `control_ms` | one observation per control step of this many ms (`brain_ms` in `config` is the same number) |
| `proprio.params` | `gain` (mV at unit input), `touch_scale`, `touch` (`[start, stop]` of the contact sensors, or null), `obs_dim` |
| `proprio.tables` | `matrix_indptr`, `matrix_indices`, `matrix_values`: a CSR matrix `n_neurons x obs_dim` |

The drive on each step: `x = obs` with `x[touch] = tanh(x[touch] * touch_scale)`, then
`drive = gain * max(matrix @ x, 0)`. Features are the readout rates times `readout_scale`,
followed by the raw observation when `config.include_proprio` is true. The observation is
what the `flybody` MuJoCo model reports for the listed observables (joint positions and
velocities, contact forces, gyro, accelerometer, ...); a consumer running the exported MuJoCo
model (`neurofly export-body --mjcf`) assembles the same vector from its own simulation.

### policy

* `type: linear`: `action = tanh(W @ features + b)`, `W` of shape `actions x features`.
* `type: mlp`: if `obs_normalized`, `x = clip((features - obs_mean) / sqrt(obs_var + obs_eps),
  -obs_clip, obs_clip)`; then `n_layers` dense layers `W{i} @ x + b{i}`, with `activation`
  (`tanh` or `relu`) after every layer except the last; `action = clip(x, -1, 1)`.

## The server protocol

`neurofly-core serve <artifact>` speaks JSON lines: one request per line on stdin, one
response per line on stdout, logs on stderr. `--ws host:port` serves the same over a
WebSocket. The first line written is `{"ok": true, "ready": true, ...info}`.

| Request | Response |
|---|---|
| `{"op": "info"}` | name, `n_neurons`, `n_features`, `n_actions`, `controls`, `layout`, `brain_ms`, `has_policy`, `has_audition`, `sample_rate`, `retina_grid`, `detection_classes` (or null) |
| `{"op": "reset"}` | `{"ok": true}` |
| `{"op": "step", "frame": b64, "width": w, "height": h, "format"?: "rgb"\|"png"\|"jpeg", "audio"?: b64, "sample_rate"?: n, "channels"?: c, "reward"?: r, "observe_only"?: bool, "detections"?: [{"class": id or name, "box": [x0, y0, x1, y1], "score"?: s}, ...]}` | `{"ok": true, "t", "spikes", "action", "held", "keys", "buttons", "dx", "dy", "scroll"}`; with `observe_only` or without a policy: `{"ok": true, "t", "spikes", "features"}` |
| `{"op": "observe", ...the step fields...}` | `{"ok": true, "t", "spikes", "features"}`: the feature vector, no policy involved |
| `{"op": "set_policy", "type": "linear", "W": [[...]], "b": [...]}` | installs a linear policy (`W` is `actions x features`); `{"ok": true, "type": "linear"}` |
| `{"op": "set_policy", "type": "mlp", "layers": [{"W", "b"}, ...], "activation"?, "obs_mean"?, "obs_var"?, "obs_clip"?, "obs_eps"?}` | installs an MLP policy |
| `{"op": "save", "path": dir, "name"?: str, "extra"?: {}}` | writes the current model as an artifact; `{"ok": true, "path"}` |
| `{"op": "body_step", "obs": [...], "reward"?: r, "observe_only"?: bool}` | body artifacts: `{"ok": true, "t", "spikes", "action"}` with one actuator command per entry, or `features` without a policy |
| `{"op": "body_observe", "obs": [...]}` | body artifacts: the feature vector |
| `{"op": "stimulate", <selection>, "mv": 20}` | adds 20 mV of drive to the selected neurons on every brain step; `{"ok": true, "n"}` |
| `{"op": "silence", <selection>}` | the selected neurons stop spiking; `{"ok": true, "n"}` |
| `{"op": "probe", <selection>}` / `{"op": "probe", "off": true}` | later `step` / `observe` replies carry `"probe": {"spikes": [...], "rates": [...]}` for those neurons |
| `{"op": "activity", "on": true, "substeps"?: bool}` | later `step` / `observe` / `body_step` replies carry `"activity": {"t", "indices", "counts", "steps"?}`: every neuron that fired during the observation and how many times (`steps`: the indices per brain step); `{"ok": true, "n"}`. Over `--ws` the sender is also subscribed: it receives every later step's activity as a pushed message, whichever client stepped |
| `{"op": "positions"}` | `{"ok": true, "n", "unit", "positions": [[x, y, z], ...] or null, "known": [...], "superclass": [...], "populations": {name: [indices]}}` |
| `{"op": "clear"}` | undoes every stimulation and silencing (the probe stays) |
| `{"op": "select", <selection>}` | `{"ok": true, "n", "indices"}` |
| `{"op": "close"}` | `{"ok": true, "bye": true}` and the process exits |

A `<selection>` is exactly one of `"indices": [...]`, `"ids": [...]` (connectome ids),
`"type_re": "^PPL1"`, `"superclass": "descending_neuron"`, or `"name"` in `readout`,
`retina`, `audition`, `punish`. Step replies for a layout with gamepad entries also carry
`"pad_buttons": [...]` and `"axes": {"lx": ..}`.

`observe`, `set_policy` and `save` are the training seam for other languages: a base
artifact (`neurofly build`) has no policy; a client reads features, trains whatever it
likes, installs the result and saves a complete artifact.

## The gRPC service

`neurofly-core serve <artifact> --grpc host:port` serves the same operations as a gRPC
service defined in `core/src/neurofly_core/rpc/neurofly.proto`: `Info`, `Reset`, `Step`,
`Observe`, `Stream` (bidirectional, one reply per request), `BodyStep` and `BodyStream`
(body artifacts), `SetPolicy`, `Save`, `Stimulate`, `Silence`, `Probe`, `Clear`, `Select`,
`Activity` and `Positions`. `StepRequest.detections` carries detected objects
(`Detection`: `class_id`, `x0`, `y0`, `x1`, `y1`, `score`) and `Info.detection_classes`
the classes an artifact expects.
`Info.kind` says which kind is loaded; `Info.n_obs` and `Info.obs_json` describe a body
artifact's observation vector. Frames and audio are `bytes` (no
base64); a linear policy is one `Layer` with `w` row-major. Generate a client for any
language with `protoc` from that file.

`frame` is base64 of raw RGB bytes (row-major, 3 bytes per pixel) unless `format` names an
encoded image. `audio` is base64 of float32 little-endian samples, interleaved if
`channels > 1`. Errors: `{"ok": false, "error": "..."}`; the session continues.

# neurofly for Python

The runtime is a Python package already: `pip install -e ../../core[pc]` (or
`neurofly-core` once published).

```python
import neurofly_core as fc

model = fc.load_model("artifacts/myapp")
model.reset()
state, info = model.step(frame, audio)     # ControlState: keys, buttons, dx, dy, scroll
features = model.observe(frame, audio)     # or just the readout, for your own policy
```

`fc.artifact.validate(path)` checks an artifact; `neurofly-core run artifacts/myapp
--window "My App"` drives the PC directly; `neurofly-core serve` is the process the other
bindings spawn.

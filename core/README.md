# neurofly-core

The runtime. Loads an artifact (a trained fly-brain controller), takes frames and sound,
returns keyboard and mouse controls. No training code, no data loading.

```
pip install -e core[pc]          # pc: screen and sound capture, keyboard and mouse
neurofly-core info  artifacts/myapp
neurofly-core serve artifacts/myapp            # JSON lines over stdio, for any language
neurofly-core run   artifacts/myapp --window "My App" --dry-run
```

```python
import neurofly_core as fc
model = fc.load_model("artifacts/myapp")
state, info = model.step(frame, audio)         # keys, buttons, dx, dy, scroll
```

The artifact format is documented in `../artifact/SPEC.md`; the server protocol in
`neurofly_core/server.py`. The training package that produces artifacts is `../training`.

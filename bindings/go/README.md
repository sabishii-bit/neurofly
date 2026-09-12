# neurofly for Go

```go
import "neurofly"

fly, err := neurofly.Spawn("artifacts/myapp")          // or neurofly.WithPython("python", ...)
r, err := fly.Step(neurofly.Step{Frame: rgb, Width: 320, Height: 240, Audio: samples, SampleRate: 16000})
// r.Keys, r.Buttons, r.DX, r.DY, r.Scroll, r.Held, r.Action, r.Spikes
fly.Close()
```

`Frame` is raw RGB bytes (row-major, three per pixel) or an encoded image with
`Format: "png"`. Apply the returned controls however your program takes input.

Training from Go: `Observe` returns the feature vector, `SetPolicyLinear` / `SetPolicyMLP`
install a policy you trained, `Save` writes a complete artifact. `example/main.go` does all
of it against a base artifact:

```
neurofly build artifacts/base --brain synthetic --keys w,a --mouse
cd bindings/go && go run ./example ../../artifacts/base ../../artifacts/from_go
```

The package spawns `neurofly-core serve <artifact>` and speaks the JSON-lines protocol in
`artifact/SPEC.md`, so the runtime must be installed in a Python on the PATH
(`pip install neurofly-core`), or use `WithPython`. Use it from another module with a
`replace` directive:

```
require neurofly v0.0.0
replace neurofly => ../neurofly-template/bindings/go
```

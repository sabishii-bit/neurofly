# neurofly for Rust

```rust
use neurofly::{Client, Step};

let mut fly = Client::spawn("artifacts/myapp")?;          // or Client::with_python("python", ...)
println!("{:?}", fly.info.controls);
let r = fly.step(&Step::rgb(&frame, 320, 240).audio(&samples, 16000, 1))?;
// r.keys, r.buttons, r.dx, r.dy, r.scroll, r.held, r.action, r.spikes
fly.close()?;
```

`Step::rgb` takes raw RGB bytes (row-major, three per pixel); `Step::encoded(bytes, "png")`
takes an encoded image. Apply the returned controls however your program takes input.

The crate spawns `neurofly-core serve <artifact>` and speaks the JSON-lines protocol in
`artifact/SPEC.md`, so the runtime must be installed in a Python on the PATH
(`pip install neurofly-core`), or use `Client::with_python`. Add it to a project with a path
dependency:

```toml
[dependencies]
neurofly = { path = "../neurofly-template/bindings/rust" }
```

A native build of the runtime (the same crate, no subprocess) is the planned replacement;
the artifact and this API stay the same.

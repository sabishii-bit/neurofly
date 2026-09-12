# neurofly documentation

A fruit fly connectome run as a spiking network, wired to two worlds: your PC (screen and
sound in, keyboard and mouse out) and a physics-simulated fly body. Training happens in
Python; the result is an artifact that runs from any language.

| Read this | When you want to |
|---|---|
| [Installation](installation.md) | requirements, the two packages and their extras, the body, GPU, the data, the bindings, known install problems |
| [Getting started](getting-started.md) | run something in five minutes once installed |
| [Concepts](concepts.md) | understand the brain, the subsets, the encoders and decoders, and what is real versus engineered |
| [The PC](pc.md) | let the brain see the screen, hear the sound, find objects with a detector, and use keyboard and mouse |
| [The body](body.md) | make the simulated fly walk, with or without the brain |
| [Training](training.md) | choose between PPO, evolution strategies, imitation and plasticity; understand run directories |
| [Runtime and artifacts](runtime.md) | export a run, run it without the training stack, use it from Node, Rust, or anything else |
| [From your own project](from-your-project.md) | step by step: neurofly as a dependency of a TypeScript, Rust, Go or Python project |
| [Hosting the brain](deploy.md) | a brain in a Docker container on a server, tokens, a brain per client, recordings and training in a container |
| [Extending](extending.md) | write a `Task`, swap a source, add an encoder or a detector backend, use the Python API directly |
| [Command reference](cli.md) | every `neurofly` and `neurofly-core` command and option |
| [Performance](performance.md) | what a step costs, the two brain backends, GPU |
| [Troubleshooting](troubleshooting.md) | things that go wrong and what they mean |

Two packages: `neurofly-core` (the runtime, in `core/`) and `neurofly-training` (everything
that builds and trains, in `training/`). Two commands: `neurofly` and `neurofly-core`; each
subcommand takes `--help`.

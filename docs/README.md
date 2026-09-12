# neurofly documentation

A fruit fly connectome run as a spiking network, wired to two worlds: your PC (screen and
sound in, keyboard and mouse out) and a physics-simulated fly body. Training happens in
Python; the result is an artifact that runs from any language.

| Read this | When you want to |
|---|---|
| [Getting started](getting-started.md) | install, fetch the data, run the tests, run something in five minutes |
| [Concepts](concepts.md) | understand the brain, the subsets, the encoders and decoders, and what is real versus engineered |
| [The PC](pc.md) | let the brain see the screen, hear the sound, and use keyboard and mouse |
| [The body](body.md) | make the simulated fly walk, with or without the brain |
| [Training](training.md) | choose between PPO, evolution strategies, imitation and plasticity; understand run directories |
| [Runtime and artifacts](runtime.md) | export a run, run it without the training stack, use it from Node, Rust, or anything else |
| [Extending](extending.md) | write a `Task`, swap a source, add an encoder, use the Python API directly |
| [Command reference](cli.md) | every `neurofly` and `neurofly-core` command and option |
| [Performance](performance.md) | what a step costs, the two brain backends, GPU |
| [Troubleshooting](troubleshooting.md) | things that go wrong and what they mean |

Two packages: `neurofly-core` (the runtime, in `core/`) and `neurofly-training` (everything
that builds and trains, in `training/`). Two commands: `neurofly` and `neurofly-core`; each
subcommand takes `--help`.

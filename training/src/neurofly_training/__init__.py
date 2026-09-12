"""neurofly-training: build and train fly-brain controllers, export them for neurofly-core.

    data/      the connectome: download, load, subsets, named neuron populations
    build.py   connectome + options -> a neurofly_core.Model (encoders built from annotations)
    body/      the MuJoCo fly: tasks, Gymnasium wrapper, proprioceptive encoder, actuator decoder
    pc/        the PC as an environment: PCEnv, Task, imitation
    envs.py    make_env() and the shared command-line options
    export.py  a run directory -> an artifact directory
    cli/       the `neurofly` command: train, es, imitate, record, play, watch, export, ...
"""
__version__ = "0.1.0"

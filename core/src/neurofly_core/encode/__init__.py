"""World to drive: encoders turn frames and sound into external drive on specific neurons.

At runtime an encoder is just tables (which neurons, which pixels or bands, what weights)
and a rule to apply them. Building the tables from a connectome's annotations is the
training package's job (``neurofly_training.build``); the tables travel in the artifact.
"""

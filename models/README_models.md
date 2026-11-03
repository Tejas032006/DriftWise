# models/

Nothing is stored in this folder on purpose.

All models in DriftWise (the frozen baselines, the candidates, and the retrained models) are small
scikit-learn classifiers that are rebuilt inside every run from a fixed random seed.
Saving them would add files without adding anything the code cannot reproduce.

If a later experiment needs a saved model (for example a larger model or a model trained once and reused),
save it here and add a line to `logs/experiment_log.md` saying which notebook produced it.

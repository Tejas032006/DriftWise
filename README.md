# DriftWise

Concept drift and safe model adaptation for network intrusion detection: an evaluation of gated model updates.

**Status: research in progress. All results are exploratory, and novelty is not claimed.**

## Question

After concept drift, when should a deployed intrusion detector update itself? We study a gated policy:
a drift alarm starts an investigation, candidate models are trained, and a candidate replaces the current model
only if validation evidence supports it. It is compared with never updating, periodic retraining, retraining at every
alarm, no-gate retraining schedules, and Adaptive Random Forest (ARF).

## Findings so far (exploratory)

- The overall error rate hides attack-class drift. Monitoring missed attacks and false alarms separately is much cleaner.
- Accepting the first validated candidate fails, because it is trained on stale data. Keeping the investigation open fixes it.
- Some conservative accept rule is necessary: a plain "looks better" rule made harmful updates in 33% of no-drift streams.
- A simple margin rule matched the sequential gate on safety and accuracy in the synthetic and UNSW-NB15 tests;
  the gate's statistics showed no measurable advantage beyond a margin.
- Gating costs far more compute and labels: about 10 times more models trained than a periodic schedule for similar accuracy.
- On UNSW-NB15, gated updates recovered attack recall in the later period (about 0.88 to about 0.97).
- On synthetic streams, ARF with a simple class-imbalance fix scored higher balanced accuracy than our gated method.

Related work (for example DriftSurf, CALIPER, certified model-update promotion, ARF) already covers the components of the
method, so this repository is best read as an evaluation study, not a new algorithm.

## Repository layout

    src/        ega_pipeline.py (experiment functions), ega_gate.py (sequential gate)
    notebooks/  Colab notebooks (see the order below)
    configs/    driftwise_settings.yaml (a record of every setting used)
    logs/       experiment_log.md (what each experiment showed)
    results/    result CSVs, plots/ (figures) and tables/
    models/     intentionally empty (models are rebuilt from fixed seeds)
    data/       NOT in this repository (see below)

## How to reproduce

1. **Environment:** Google Colab. Install once, then restart the session: `pip install river "numpy<2.3"`.
2. **Data (not included):** download UNSW-NB15 from the official UNSW Canberra page, put `UNSW-NB15_1.csv` in `data/raw/`,
   and run the preprocessing notebook. It creates the cleaned, chronologically sorted file in `data/processed/`.
   The real-data notebooks need that file.
3. **Notebooks, in this order:** preprocessing, `DriftWise_Main_clean` (synthetic experiments),
   `DriftWise_Real_Data_Experiments`, the ablation, sweep and ARF notebooks, then `DriftWise_Figures`.
4. Functions are in `src/ega_pipeline.py`. In a notebook:
   `sys.path.insert(0, "<path to src>")` then `from ega_pipeline import *`.

## Limitations

- UNSW-NB15 has attacks only in the first part of its recording, and the drift in it is mild.
- Real-data results come from one fixed sequence and 2 to 3 random seeds (model randomness only).
- ARF was run on synthetic streams only.
- The literature scan was a first pass.

## License and data

No license has been chosen yet. The UNSW-NB15 dataset has its own terms and is not redistributed here.

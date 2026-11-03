# DriftWise experiment log

A short record of each experiment: which notebook ran it, which result file holds the numbers, and what it showed.
Numbers are the ones printed when the experiment was run. Findings are exploratory unless stated.
Result files are in `results/`; figures in `results/plots/`; tables in `results/tables/`.

## Data
| Step | Notebook | Finding |
|---|---|---|
| UNSW-NB15 audit and cleaning | `DriftWise_UNSW_Data_Preprocessing_clean` | 59,213 exact duplicates removed (700,001 to 640,788 rows; attacks 22,215 to 14,278). All attacks occur in the first 173,433 rows; the last 467,355 rows are normal-only. A chronological train/test split would have no attacks in the test set. |

## Synthetic streams (`DriftWise_Main_clean`, functions in `src/ega_pipeline.py`)
| Experiment | Result file | Finding |
|---|---|---|
| Frozen model under pattern drift | (printed) | Balanced accuracy fell from about 0.84 to 0.54. Overall error rose only moderately while attack recall collapsed. |
| First EGA design (stop at first accepted candidate) | `ega_v2_fresh_seeds_pattern_and_rate_only.csv` | Worse than retraining at every alarm: -0.061 [-0.104, -0.017] balanced accuracy over 30 fresh seeds. Cause: the first accepted candidate was trained on pre-drift data. |
| ega_v2 (keep investigating after an accept) | `ega_v2_fresh_seeds_pattern_and_rate_only.csv` | +0.029 [+0.009, +0.048] balanced accuracy and +0.064 [+0.021, +0.106] worst-class recall vs retrain once at the alarm. |
| No-gate retraining schedules | `ega_v2_vs_schedules_pattern_lag0.csv` | ega_v2 matches the schedule that retrains at the alarm, +300 and +600 (+0.001 [-0.014, +0.015]). |
| Single delayed retrain | `single_delayed_retrain_pattern_lag0.csv` | ega_v2 matches a single retrain 300 records after the alarm (+0.011 [-0.003, +0.024]) with more updates. |
| Gradual drift and delayed labels | `ega_v2_vs_baselines_gradual_and_delayed.csv` | ega_v2 matched the no-gate schedule with fewer updates in all three settings tested. |
| Cost | `ega_v2_cost_comparison.csv` | ega_v2 deploys about 1.65 models but trains about 16.6 and uses about 6,240 validation labels (label uses, with overlapping blocks); the schedules train 2 to 4 models. |
| Gate ablation and trade-off sweep | `ablation_gate_*.csv`, `gate_sweep_*.csv`, `gate_tradeoff_summary.csv` | On streams with no drift, a plain "looks better" rule made a harmful update in 33% of streams; the sequential gate in 1%. A simple rule with a 0.05 margin was as safe as the gate, with similar accuracy. |
| ARF baseline (synthetic only) | `arf_baseline_comparison.csv` | ARF with attacks shown 3 times scored about 0.024 higher balanced accuracy than ega_v2; default ARF matched it on balanced accuracy but missed far more attacks (worst-class recall 0.595 vs 0.732). |

## Real data, UNSW-NB15 (`DriftWise_Real_Data_Experiments`, `DriftWise_Real_Gate_Ablation`)
| Experiment | Result file | Finding |
|---|---|---|
| Frozen model by block | (printed) | Balanced accuracy 0.95 to 0.99 early; attack recall falls to about 0.80 to 0.95 after about row 117,000. A mild, natural drift. |
| ADWIN on overall error vs per class | (printed) | The overall detector raised 38 alarms. A detector on missed attacks raised 3 (2 upward). Overall error fell while attack recall got worse. |
| Periodic retraining headroom | `real_data_periodic_headroom.csv` | Retraining helped in the later period (+0.012 to +0.029) but one schedule was 0.046 worse overall than never updating. |
| Methods compared and stability | `real_data_ega_vs_baselines.csv`, `real_data_stability.csv` | ega_v2 matched periodic retraining with about half the updates (7 vs 15). 3 random seeds, one fixed stream. |
| Gate ablation with hindsight check | `real_data_gate_ablation.csv`, `real_data_gate_ablation_update_effects.csv` | All gated variants were indistinguishable on accuracy. Hindsight: 0 of 14 updates harmful for the sequential gate, 3 of 43 for the plain rule. Too few to separate the rules statistically. All gated variants trained about 150 models vs 16 for periodic. |

## Limits to state in the paper
- UNSW-NB15 has attacks only in the first part of the recording, and the drift in it is mild.
- Real-data results come from one long sequence and 2 to 3 random seeds (model randomness only).
- ARF was run on synthetic streams only.
- The drift zone (row 117,000 onward) and the harmful threshold (0.01) were chosen by the author.
- The literature scan was a first pass; novelty is not established.

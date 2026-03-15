This repair fixes the first runtime bug observed on Narval and improves comparison fairness.

Changed files:
- src/fr3_twc/pipeline.py
- src/fr3_twc/unfolded.py

What changed:
- Fixed Keras call usage for the unfolded model by passing WidebandBatch as a keyword argument (`model(batch=..., training=...)`) in both training and inference.
- Fixed weight-sensitivity evaluation so all weight profiles are compared on the same underlying random geometry within each Monte Carlo sample.
- Fixed legacy-summary ingestion so all sweep points are imported into `metrics_summary_with_legacy.csv`, legacy algorithm names are normalized (timestamps removed), and the available FS interference metric is used consistently.

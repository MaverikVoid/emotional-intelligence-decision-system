# ArvyaX Reflective State Prediction

This submission builds a local end-to-end pipeline for emotional understanding, intensity prediction, decision support, and uncertainty awareness on short post-session reflections.

## Files

- `code.ipynb`: executed notebook with EDA, model comparisons, graphs, commentary, and final export steps.
- `arvyax_solution.py`: reusable training and inference pipeline.
- `build_notebook.py`: recreates the executed notebook.
- `predictions.csv`: final test-set predictions in the required format.
- `ERROR_ANALYSIS.md`: ten curated failure cases and fixes.
- `EDGE_PLAN.md`: mobile/on-device deployment plan and tradeoffs.

## Approach

- Emotional state is modeled as multi-class text classification.
- The final state model is a text-only `LinearSVC` with word and character TF-IDF features.
- Intensity is treated as ordinal regression rather than plain classification because adjacent labels are fuzzy and noisy.
- The final intensity model is an `XGBRegressor` over text plus metadata features, then rounded to the nearest integer from 1 to 5.
- Confidence is built from state margin strength, intensity rounding stability, short-text penalties, missing metadata penalties, and contradiction penalties.
- The decision engine is rule-based and uses predicted state, predicted intensity, stress, energy, sleep, and time of day to choose `what_to_do` and `when_to_do`.

## Validation Highlights

- Emotional state CV accuracy: `0.682`
- Intensity exact-match accuracy: `0.218`
- Intensity within-1 accuracy: `0.608`
- Intensity MAE after rounding: `1.240`
- Uncertain flag rate: `9.4%`
- State error rate inside the uncertain slice: `84.1%`

## Main Findings

- Text carries the strongest signal by far.
- Metadata is useful for decision-making and edge handling, but it was too weak to beat the text-only state model.
- Intensity labels are much noisier than emotional-state labels.
- Very short repeated reflections like `okay session`, `it was fine`, and `felt better` create genuine ambiguity because the same text appears with multiple different labels.

## How To Run

1. Install local dependencies if needed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `nbformat`, `nbclient`.
2. Generate the required submission file:

```bash
python arvyax_solution.py
```

3. Rebuild the full executed notebook:

```bash
python build_notebook.py
```

## Output Format

`predictions.csv` contains:

- `id`
- `predicted_state`
- `predicted_intensity`
- `confidence`
- `uncertain_flag`
- `what_to_do`
- `when_to_do`

The pipeline also creates an optional `supportive_message` internally, but I kept the exported CSV aligned with the assignment format.

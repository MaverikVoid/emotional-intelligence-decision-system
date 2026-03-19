# ArvyaX Reflective State Prediction

This project builds a **context-aware emotional intelligence system** that goes beyond prediction.

It is designed to:
- understand noisy and ambiguous human reflections  
- infer emotional state and intensity  
- decide meaningful next actions  
- estimate its own uncertainty  

The system integrates **text understanding, contextual reasoning, decision logic, and uncertainty awareness** into a unified pipeline.

---

## System Architecture

Input (text + metadata)  
→ Feature Extraction (TF-IDF + structured features)  
→ Emotional State Model (LinearSVC)  
→ Intensity Model (XGBoost Regressor)  
→ Decision Engine (rule-based reasoning)  
→ Uncertainty Layer (confidence + flags)  
→ Final Output (state, intensity, action, timing)

---

## Files

- `analysis.ipynb`: executed notebook with EDA, model comparisons, graphs, and insights  
- `pipeline.py`: end-to-end training + inference pipeline  
- `build_notebook.py`: recreates the executed notebook  
- `predictions.csv`: final predictions in required format  
- `ERROR_ANALYSIS.md`: ten curated failure cases with explanations and improvements  
- `EDGE_PLAN.md`: mobile/on-device deployment plan and tradeoffs  

---

## Approach

### Emotional State Modeling
Modeled as a multi-class classification problem.  
A **LinearSVC with word + character TF-IDF** was chosen because:
- performs strongly on short, noisy text  
- handles high-dimensional sparse features efficiently  
- reduces overfitting risk on small datasets  

---

### Intensity Modeling
Treated as an **ordinal regression problem** rather than plain classification because:
- labels are ordered (1–5)  
- adjacent labels are often ambiguous  
- regression captures smoother transitions  

The final model is an **XGBoost Regressor** over text + metadata features, with predictions rounded to [1–5].

---

### Decision Engine
A **rule-based reasoning system** that uses:
- predicted state  
- predicted intensity  
- stress, energy, sleep  
- time of day  

to generate:
- `what_to_do` (action)  
- `when_to_do` (timing)  

---

### Uncertainty Modeling
Confidence is computed using:
- model confidence (margin / prediction stability)  
- input quality (text length, missing data)  
- conflicting signals (e.g., high energy + high stress)  

This enables the system to detect when predictions are unreliable.

---

## Validation Highlights

- Emotional state CV accuracy: `0.682`  
- Intensity exact-match accuracy: `0.218`  
- Intensity within-1 accuracy: `0.608`  
- Intensity MAE after rounding: `1.240`  
- Uncertain flag rate: `9.4%`  
- State error rate inside the uncertain slice: `84.1%`  

---

## Ablation Study

| Model | Features | Observation |
|------|--------|------------|
| Text-only (LinearSVC) | TF-IDF | Strong baseline performance |
| Text + Metadata | TF-IDF + structured | Improved intensity prediction, limited impact on state |

**Conclusion:**  
Text is the dominant signal for emotional understanding, while metadata improves contextual reasoning and decision-making.

---

## Key Insights

- Text carries the strongest signal for emotional state prediction.  
- Metadata provides useful context but is weaker than text alone.  
- Intensity labels are significantly noisier than emotional-state labels.  
- Very short reflections introduce ambiguity and reduce prediction reliability.  
- The uncertainty mechanism effectively identifies difficult cases, with high error concentration in low-confidence predictions.  

---

## Why This Matters

Real-world AI systems must operate under **uncertainty, noise, and incomplete information**.  

This system demonstrates how:
- prediction  
- reasoning  
- uncertainty awareness  

can be combined to **guide users toward better mental states**, rather than simply labeling them.

---

## How To Run

1. Install dependencies:
```bash
pip install -r requirements.txt

## How To Run

1. Install local dependencies if needed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `nbformat`, `nbclient`.
2. Generate the required submission file:

```bash
python pipeline.py
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

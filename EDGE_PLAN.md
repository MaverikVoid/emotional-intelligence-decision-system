# Edge / Offline Plan

## Goal

Run the reflective-state system locally on a phone without calling any hosted API, while keeping latency low enough for a smooth post-session experience.

## Practical Mobile Design

- Keep the emotional-state model as a linear text classifier. TF-IDF plus a linear head is small, fast, and easy to export.
- Replace the notebook pipeline with a compact inference bundle that stores:
  - the TF-IDF vocabulary
  - the linear state weights
  - a lightweight intensity model
  - the rule-based decision engine
- Run preprocessing, prediction, uncertainty scoring, and decision logic fully on-device.

## Recommended On-Device Version

- State model: hashed TF-IDF or a pruned vocabulary plus linear SVM/logistic model.
- Intensity model: a lighter ridge regressor or shallow gradient-boosted model if memory is tight.
- Decision layer: pure rules in app code.
- Supportive message: template-based generation instead of a generative model.

## Why This Fits Edge Deployment

- Sparse text features are cheap on CPU.
- Linear models have tiny inference cost compared with transformer models.
- The decision layer is deterministic and near-zero latency.
- The uncertainty flag lets the app safely fall back when the input is too weak.

## Latency / Size Tradeoffs

- Best accuracy in this notebook uses a richer feature stack and XGBoost for intensity, which is still local but heavier than a pure linear setup.
- For mobile, I would likely keep the state model close to the current version and simplify intensity if package size becomes a concern.
- A fully pruned linear-only version would probably lose some intensity quality, but it would be easier to ship and maintain.
- My expectation is that a linear state model plus simple regressor should run in well under a second on a modern phone CPU. This is an engineering estimate, not a measured benchmark.

## Robustness On Device

- Very short text like `ok` or `fine`: lower confidence, set `uncertain_flag = 1`, and default to a safe gentle action such as `pause` or `journaling`.
- Missing values: use the same train-time imputers and missing-category tokens as the notebook pipeline.
- Contradictory signals: keep the contradiction penalty in confidence scoring so the app does not over-commit when text and metadata disagree.
- Offline mode: all features are local, so no network is needed after the model bundle is shipped.

## Deployment Path

1. Freeze the preprocessing vocabulary and trained weights.
2. Export the final lightweight models to ONNX or another mobile-friendly format.
3. Reimplement the decision rules in the app layer.
4. Log only anonymous local telemetry or opt-in feedback if product policy allows.
5. Use the uncertainty flag to trigger safer UX flows instead of overconfident advice.

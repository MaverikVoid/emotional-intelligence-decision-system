# Error Analysis

The hardest cases were not random. Most failures came from short generic text, contradictory cues, and label noise where the same reflection appeared under different targets.

| Row | ID | Reflection | Actual | Predicted | What went wrong | How to improve |
|---|---:|---|---|---|---|---|
| 1098 | 979 | `okay session` | `mixed`, intensity `1` | `focused`, intensity `4` | Two-word text with almost no semantic anchor. Metadata says high stress and conflicted quality, but the lexical signal is too weak. | Add an abstain rule for ultra-short text and route to a fallback metadata-only model or user clarification. |
| 839 | 840 | `honestly not much change` | `overwhelmed`, intensity `5` | `focused`, intensity `2` | The phrase sounds neutral on the surface, but metadata shows high stress, low energy, and night-time fatigue. | Use stronger contradiction-aware fusion so high-stress / low-energy context can override calm-looking text. |
| 549 | 550 | `okay session` | `neutral`, intensity `1` | `focused`, intensity `3` | Same repeated phrase appears with several labels in train, so this is a label-noise hotspot. | Group repeated generic texts and smooth their labels with metadata-conditioned priors. |
| 789 | 790 | `it was fine` | `mixed`, intensity `4` | `focused`, intensity `3` | The wording is vague, but stress is high and energy is low. The model trusts the bland text too much. | Add more explicit vague-text detection and lower confidence faster when the reflection is generic. |
| 917 | 798 | `tired but okay` | `mixed`, intensity `3` | `focused`, intensity `4` | The text contains both relief and fatigue, which is a classic transition-state sample. | Add a dedicated transition-state feature family for contrast phrases like `but okay`, `a bit better`, and `still tired`. |
| 672 | 673 | `it was fine` | `restless`, intensity `1` | `focused`, intensity `3` | Another repeated generic text with conflicting labels across the dataset. | Reduce overfitting to generic positive phrases and add duplicate-text ambiguity penalties during training. |
| 1088 | 969 | `bit restless` | `calm`, intensity `5` | `restless`, intensity `2` | The literal phrase strongly points toward `restless`, but the gold label says `calm`. This looks like either label noise or a context-heavy edge case. | Review labels manually or train with noise-robust loss / confidence-based relabeling. |
| 1151 | 1032 | `bit restless` | `calm`, intensity `4` | `restless`, intensity `2` | Same issue as above: the text says one thing and the label says another. | Add a human review pass for duplicated short texts with unstable labels. |
| 1187 | 1068 | `felt better` | `mixed`, intensity `3` | `overwhelmed`, intensity `2` | The reflection suggests improvement but does not say whether the user is fully regulated or only slightly better. | Add temporal transition features such as `better`, `lighter`, `back to normal`, and combine them with stress trend features if available. |
| 542 | 543 | `mind was all over the place` | `mixed`, intensity `3` | `calm`, intensity `4` | The text is clearly activated, but reflection quality is conflicted and the model still under-read the stress cue. | Add phrase-level features for racing-thought patterns and train a stronger contradiction detector. |

## Cross-Case Patterns

- Very short text is the biggest failure source.
- Repeated generic reflections create unstable supervision because the same wording appears under different labels.
- Intensity is noisier than emotional state, so exact-match scoring understates the model's usefulness.
- Contradictory inputs matter: calm-looking text can still hide high stress, fatigue, or conflict.

## Next Improvements

- Add weak-label denoising for repeated low-information reflections.
- Train a separate short-text specialist model.
- Use ordinal loss or pairwise ranking for intensity instead of plain rounded regression.
- Add a contradiction score as an explicit feature instead of only using it in uncertainty.

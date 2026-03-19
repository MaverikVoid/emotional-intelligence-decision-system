from pathlib import Path
import os
import textwrap

import nbformat as nbf
import jupyter_core.paths
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "code.ipynb"
RUNTIME_DIR = ROOT / ".jupyter_runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ["JUPYTER_RUNTIME_DIR"] = str(RUNTIME_DIR)
jupyter_core.paths.win32_restrict_file_to_user = lambda fname: None


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # ArvyaX Internship Assignment

        This notebook rebuilds the full pipeline from scratch with a stronger validation setup, explicit uncertainty handling, a rule-based decision layer, and a short deployment plan mindset.
        """
    ),
    md(
        """
        ## 1. Imports And Setup

        I kept the notebook readable and modular by putting the reusable training logic in `arvyax_solution.py`, then using this notebook for analysis, plots, commentary, and final artifact generation.
        """
    ),
    code(
        """
        import warnings
        warnings.filterwarnings("ignore")

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from IPython.display import display
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        from arvyax_solution import (
            TextColumnSelector,
            add_manual_features,
            build_state_pipeline,
            build_training_summary,
            build_supportive_message,
            decide_action,
            failure_case_table,
            fit_final_models,
            load_datasets,
            predict_submission,
            save_predictions_csv,
            state_feature_importance_table,
        )

        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["figure.figsize"] = (10, 5)
        pd.set_option("display.max_colwidth", 120)
        RANDOM_STATE = 42
        """
    ),
    md("## 2. Load Data"),
    code(
        """
        train_df, test_df = load_datasets()
        train_enriched = add_manual_features(train_df)
        test_enriched = add_manual_features(test_df)

        overview = pd.DataFrame(
            {
                "dataset": ["train", "test"],
                "rows": [len(train_df), len(test_df)],
                "columns": [train_df.shape[1], test_df.shape[1]],
                "unique_ids": [train_df["id"].nunique(), test_df["id"].nunique()],
                "duplicate_id_rows": [len(train_df) - train_df["id"].nunique(), len(test_df) - test_df["id"].nunique()],
            }
        )
        display(overview)
        display(train_df.head(3))
        """
    ),
    code(
        """
        print("Quick takeaways:")
        print(f"- The training set has {len(train_df):,} rows and the test set has {len(test_df):,} rows.")
        print(f"- Train IDs are not unique: {len(train_df) - train_df['id'].nunique()} rows share an existing id, so I use a row-level key internally.")
        print(f"- The reflections are short: median length is {train_enriched['text_word_count'].median():.0f} words.")
        """
    ),
    md("## 3. Data Quality Checks"),
    code(
        """
        missing_summary = pd.DataFrame(
            {
                "train_missing": train_df.isna().sum(),
                "test_missing": test_df.isna().sum(),
            }
        ).sort_values("train_missing", ascending=False)
        display(missing_summary)

        duplicate_texts = train_df["journal_text"].duplicated().sum()
        duplicate_rows = train_df.duplicated().sum()
        print(f"Duplicate journal texts: {duplicate_texts}")
        print(f"Fully duplicate rows: {duplicate_rows}")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.countplot(data=train_df, x="emotional_state", order=train_df["emotional_state"].value_counts().index, ax=axes[0])
        axes[0].set_title("Emotional State Distribution")
        axes[0].tick_params(axis="x", rotation=25)

        sns.countplot(data=train_df, x="intensity", ax=axes[1])
        axes[1].set_title("Intensity Distribution")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        print("Insights from the data quality checks:")
        print("- The emotional state classes are nicely balanced, so low accuracy was not caused by class imbalance.")
        print("- Missingness is concentrated in `face_emotion_hint`, `previous_day_mood`, and a few `sleep_hours` values.")
        print(f"- There are {duplicate_texts} repeated reflections, and many of them map to different labels. That is a real source of label noise.")
        """
    ),
    md("## 4. Exploratory Analysis"),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.boxplot(data=train_enriched, x="emotional_state", y="text_word_count", ax=axes[0])
        axes[0].set_title("Word Count By Emotional State")
        axes[0].tick_params(axis="x", rotation=25)

        heatmap_data = pd.crosstab(train_df["stress_level"], train_df["emotional_state"], normalize="index")
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
        axes[1].set_title("State Mix Within Each Stress Level")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        print("EDA takeaways:")
        print(f"- Mixed reflections are the longest on average at {train_enriched.groupby('emotional_state')['text_word_count'].mean()['mixed']:.1f} words.")
        print("- Stress level helps, but it is not enough on its own. Even at high stress, several emotional states remain plausible.")
        print("- This explains why text carries the main signal and metadata mostly acts as a weak context feature.")
        """
    ),
    md("## 5. Cross-Validated Modeling"),
    code(
        """
        oof_df, summary = build_training_summary(train_df)

        state_metrics = pd.DataFrame([summary.state_metrics]).T.rename(columns={0: "value"})
        intensity_metrics = pd.DataFrame([summary.intensity_metrics]).T.rename(columns={0: "value"})
        state_ablation = pd.DataFrame(summary.state_ablation.items(), columns=["model", "cv_accuracy"])
        intensity_ablation = pd.DataFrame(summary.intensity_ablation.items(), columns=["model", "cv_mae"])

        display(state_metrics)
        display(intensity_metrics)
        display(state_ablation.sort_values("cv_accuracy", ascending=False))
        display(intensity_ablation.sort_values("cv_mae"))
        """
    ),
    code(
        """
        print("Modeling commentary:")
        print(f"- Final emotional-state model: text-only LinearSVC with word + character TF-IDF, CV accuracy = {summary.state_metrics['state_accuracy']:.3f}.")
        print(f"- Text-only state model beat text+metadata ({summary.state_ablation['text_only_svm']:.3f} vs {summary.state_ablation['text_plus_metadata']:.3f}).")
        print(f"- Metadata-only state accuracy was just {summary.state_ablation['metadata_only']:.3f}, so metadata alone is too weak.")
        print(f"- Intensity is much noisier. Exact accuracy is {summary.intensity_metrics['intensity_exact_accuracy']:.3f}, but within-1 accuracy improves to {summary.intensity_metrics['intensity_within_1_accuracy']:.3f}.")
        print("- I treated intensity as ordinal regression in the final system because the gap between nearby intensity levels is fuzzy.")
        """
    ),
    md("## 6. Confusion Matrix"),
    code(
        """
        confusion = pd.crosstab(
            oof_df["emotional_state"],
            oof_df["state_pred"],
            normalize="index"
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt=".2f", cmap="rocket_r")
        plt.title("Out-of-Fold State Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()
        """
    ),
    code(
        """
        print("Confusion matrix commentary:")
        print("- The main confusions happen among `mixed`, `neutral`, and `focused`, which is expected because many reflections describe transitions rather than a single pure state.")
        print("- `Overwhelmed` and `restless` also overlap whenever the text emphasizes activation without a clear emotional direction.")
        """
    ),
    md("## 7. Uncertainty Analysis"),
    code(
        """
        confidence_buckets = (
            oof_df.assign(
                confidence_bucket=pd.cut(
                    oof_df["combined_confidence"],
                    bins=[0, 0.4, 0.55, 0.7, 0.85, 1.0],
                    include_lowest=True,
                )
            )
            .groupby("confidence_bucket")[["state_correct", "intensity_correct"]]
            .mean()
            .reset_index()
        )
        display(confidence_buckets)

        plt.figure(figsize=(9, 4))
        sns.barplot(data=confidence_buckets, x="confidence_bucket", y="state_correct", color="#4C72B0")
        plt.ylim(0, 1)
        plt.title("State Accuracy By Confidence Bucket")
        plt.ylabel("Accuracy")
        plt.xlabel("Confidence bucket")
        plt.show()
        """
    ),
    code(
        """
        uncertain_error_rate = summary.state_metrics["state_error_rate_when_uncertain"]
        print("Uncertainty commentary:")
        print(f"- The system flags about {summary.state_metrics['state_uncertainty_rate']:.1%} of samples as uncertain.")
        print(f"- Inside that flagged slice, the state error rate jumps to {uncertain_error_rate:.1%}, so the flag is concentrated on genuinely difficult cases.")
        print("- Confidence falls sharply for short generic reflections like `okay session` or `it was fine`, which is exactly what we want.")
        """
    ),
    md("## 8. Feature Understanding"),
    code(
        """
        explainability_model = Pipeline(
            [
                ("text", TextColumnSelector("journal_text")),
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=3.0, random_state=RANDOM_STATE)),
            ]
        )
        explainability_model.fit(train_enriched, train_enriched["emotional_state"])

        feature_names = explainability_model.named_steps["tfidf"].get_feature_names_out()
        coef = explainability_model.named_steps["clf"].coef_
        classes = explainability_model.named_steps["clf"].classes_

        explain_rows = []
        for class_name, class_weights in zip(classes, coef):
            top_idx = np.argsort(class_weights)[-8:][::-1]
            explain_rows.append(
                pd.DataFrame(
                    {
                        "emotional_state": class_name,
                        "top_terms": [feature_names[i] for i in top_idx],
                    }
                )
            )

        explainability_terms = pd.concat(explain_rows, ignore_index=True)
        display(explainability_terms)
        """
    ),
    code(
        """
        print("Feature understanding commentary:")
        print("- Calm language is tied to words like `lighter`, `quiet`, and `my breathing`.")
        print("- Focused language leans toward `organized`, `clearer`, `concentrate`, and `planning`.")
        print("- Mixed reflections are dominated by contrast terms like `but`, `in between`, and `but still`.")
        print("- Overwhelmed and restless states are signaled by tokens such as `drained`, `flooded`, `distracted`, `tasks`, and `kept`.")
        """
    ),
    md("## 9. Decision Layer"),
    code(
        """
        decision_preview = oof_df[[
            "id",
            "journal_text",
            "state_pred",
            "intensity_pred",
            "stress_level",
            "energy_level",
            "time_of_day",
        ]].head(8).copy()

        actions = []
        times = []
        messages = []
        for _, row in decision_preview.iterrows():
            action, when = decide_action(row, row["state_pred"], int(row["intensity_pred"]))
            actions.append(action)
            times.append(when)
            messages.append(build_supportive_message(row["state_pred"], int(row["intensity_pred"]), action, when))

        decision_preview["what_to_do"] = actions
        decision_preview["when_to_do"] = times
        decision_preview["supportive_message"] = messages
        display(decision_preview)
        """
    ),
    code(
        """
        print("Decision-layer commentary:")
        print("- I kept the decision engine rule-based on purpose so it stays interpretable during the interview.")
        print("- The rules use predicted state and intensity first, then use stress, energy, sleep, and time of day to decide both `what` and `when`.")
        print("- This makes the system more product-ready than a plain classifier because it outputs an actual next action.")
        """
    ),
    md("## 10. Error Analysis"),
    code(
        """
        failure_cases = failure_case_table(oof_df, top_n=10)
        display(failure_cases)
        """
    ),
    code(
        """
        reason_breakdown = failure_cases["failure_reason"].value_counts()
        display(reason_breakdown)

        print("Error analysis commentary:")
        print("- The most common failures come from extremely short text, especially generic reflections that appear with multiple different labels.")
        print("- Some mistakes are caused by conflicting signals, for example calm-ish text combined with high stress and low energy.")
        print("- Intensity is especially noisy because repeated texts often appear under very different intensity levels.")
        """
    ),
    md("## 11. Train Final Models And Export Predictions"),
    code(
        """
        final_predictions = predict_submission(train_df, test_df)
        final_predictions.head(10)
        """
    ),
    code(
        """
        submission = save_predictions_csv()
        print("Saved predictions.csv with columns:")
        print(list(pd.read_csv("predictions.csv").columns))
        print()
        print("Sample submission rows:")
        display(pd.read_csv("predictions.csv").head(10))
        """
    ),
    md(
        """
        ## 12. Final Notes

        The strongest part of this solution is the state model plus uncertainty-aware decision engine. The hardest part of the assignment is the noisy and weakly separable intensity label, so I handled it with ordinal regression and transparent error analysis instead of pretending the signal is cleaner than it really is.
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")

client = NotebookClient(nb, timeout=900, kernel_name="python3")
client.execute()

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote executed notebook to {NOTEBOOK_PATH}")

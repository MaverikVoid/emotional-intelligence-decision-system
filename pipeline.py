from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBRegressor


TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("test.csv")

TEXT_COL = "journal_text"
STATE_COL = "emotional_state"
INTENSITY_COL = "intensity"

CAT_COLS = [
    "ambience_type",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
]
NUM_COLS = ["duration_min", "sleep_hours", "energy_level", "stress_level"]


class TextColumnSelector(BaseEstimator, TransformerMixin):
    """Select a single text column from a dataframe."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].fillna("").astype(str)


def load_datasets(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["_row_id"] = np.arange(len(enriched))
    enriched[TEXT_COL] = enriched[TEXT_COL].fillna("").astype(str)
    enriched["text_char_count"] = enriched[TEXT_COL].str.len()
    enriched["text_word_count"] = enriched[TEXT_COL].str.split().str.len()
    enriched["text_sentence_count"] = (
        enriched[TEXT_COL].str.count(r"[.!?]").clip(lower=1)
    )
    enriched["ellipsis_count"] = enriched[TEXT_COL].str.count(r"\.\.\.")
    enriched["question_count"] = enriched[TEXT_COL].str.count(r"\?")
    return enriched


def build_state_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(4, 6),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_state_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("text", TextColumnSelector(TEXT_COL)),
            ("vectorizer", build_state_vectorizer()),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_metadata_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "text_word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                TEXT_COL,
            ),
            (
                "text_char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                ),
                TEXT_COL,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUM_COLS + ["text_char_count", "text_word_count", "text_sentence_count", "ellipsis_count", "question_count"],
            ),
        ],
        sparse_threshold=0.3,
    )


def build_intensity_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", build_metadata_preprocessor()),
            ("svd", TruncatedSVD(n_components=150, random_state=42)),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            ),
        ]
    )


def build_state_ablation_with_metadata() -> Pipeline:
    return Pipeline(
        [
            ("features", build_metadata_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=3.0,
                    random_state=42,
                ),
            ),
        ]
    )


def build_state_metadata_only() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_intensity_metadata_only() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            ),
        ]
    )
    return Pipeline([("features", preprocessor), ("regressor", Ridge(alpha=1.0))])


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(centered)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def svm_confidence_from_margins(model: Pipeline, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    margins = model.decision_function(df)
    sorted_idx = np.argsort(margins, axis=1)
    top_idx = sorted_idx[:, -1]
    runner_up_idx = sorted_idx[:, -2]
    margin_gap = margins[np.arange(len(margins)), top_idx] - margins[np.arange(len(margins)), runner_up_idx]
    confidences = 1.0 / (1.0 + np.exp(-(1.3 * margin_gap - 0.5)))
    predictions = model.named_steps["classifier"].classes_[top_idx]
    return predictions, confidences


def intensity_confidence_from_raw(raw_predictions: np.ndarray) -> np.ndarray:
    rounded = np.clip(np.rint(raw_predictions), 1, 5)
    distance_from_center = np.abs(raw_predictions - rounded)
    confidence = 0.55 + 0.45 * (1.0 - np.clip(distance_from_center / 0.5, 0.0, 1.0))
    confidence = np.where((raw_predictions < 1.0) | (raw_predictions > 5.0), confidence * 0.7, confidence)
    return confidence


def data_quality_penalty(df: pd.DataFrame) -> np.ndarray:
    word_count = df[TEXT_COL].fillna("").str.split().str.len().to_numpy()
    missing_cat_count = df[CAT_COLS].isna().sum(axis=1).to_numpy()
    penalties = np.zeros(len(df), dtype=float)
    penalties += np.where(word_count <= 2, 0.12, 0.0)
    penalties += np.where(word_count <= 5, 0.05, 0.0)
    penalties += np.clip(missing_cat_count * 0.03, 0.0, 0.12)
    return penalties


def contradiction_penalty(df: pd.DataFrame, predicted_state: Iterable[str], predicted_intensity: Iterable[int]) -> np.ndarray:
    state_array = np.asarray(list(predicted_state))
    intensity_array = np.asarray(list(predicted_intensity))
    penalties = np.zeros(len(df), dtype=float)

    high_stress = df["stress_level"].to_numpy() >= 4
    low_energy = df["energy_level"].to_numpy() <= 2
    night_time = df["time_of_day"].isin(["night", "evening"]).to_numpy()

    calm_like = np.isin(state_array, ["calm", "focused"])
    activated = np.isin(state_array, ["restless", "overwhelmed"])

    penalties += np.where(high_stress & calm_like, 0.10, 0.0)
    penalties += np.where(low_energy & (state_array == "focused"), 0.08, 0.0)
    penalties += np.where(night_time & (intensity_array >= 4) & calm_like, 0.05, 0.0)
    penalties += np.where((~high_stress) & activated & (intensity_array <= 2), 0.05, 0.0)
    return penalties


def combine_confidence(
    df: pd.DataFrame,
    state_confidence: np.ndarray,
    intensity_confidence: np.ndarray,
    predicted_state: Iterable[str],
    predicted_intensity: Iterable[int],
) -> np.ndarray:
    confidence = 0.65 * state_confidence + 0.35 * intensity_confidence
    confidence -= data_quality_penalty(df)
    confidence -= contradiction_penalty(df, predicted_state, predicted_intensity)
    return np.clip(confidence, 0.01, 0.99)


def is_uncertain(df: pd.DataFrame, confidence: np.ndarray) -> np.ndarray:
    word_count = df[TEXT_COL].fillna("").str.split().str.len().to_numpy()
    missing_cat_count = df[CAT_COLS].isna().sum(axis=1).to_numpy()
    return ((confidence < 0.45) | (word_count <= 2) | (missing_cat_count >= 2)).astype(int)


def decide_action(row: pd.Series, predicted_state: str, predicted_intensity: int) -> Tuple[str, str]:
    stress = row["stress_level"]
    energy = row["energy_level"]
    time_of_day = row["time_of_day"]
    sleep_hours = row.get("sleep_hours", np.nan)
    reflection_quality = row.get("reflection_quality", "clear")

    late_day = time_of_day in {"evening", "night"}
    early_day = time_of_day in {"early_morning", "morning"}
    low_sleep = pd.notna(sleep_hours) and sleep_hours < 5.5

    if predicted_state == "focused":
        if late_day and (low_sleep or predicted_intensity >= 4):
            return "light_planning", "tomorrow_morning"
        return ("deep_work", "now") if energy >= 3 else ("light_planning", "within_15_min")

    if predicted_state == "calm":
        if late_day and (energy <= 2 or low_sleep):
            return "rest", "tonight"
        return ("yoga", "within_15_min") if energy >= 3 else ("rest", "later_today")

    if predicted_state == "overwhelmed":
        if stress >= 4 or predicted_intensity >= 4:
            return "box_breathing", "now"
        return "grounding", "within_15_min"

    if predicted_state == "restless":
        if energy >= 3:
            return "movement", "now"
        return ("box_breathing", "now") if stress >= 4 else ("grounding", "within_15_min")

    if predicted_state == "mixed":
        if reflection_quality == "conflicted" or stress >= 3:
            return "journaling", "within_15_min"
        return ("light_planning", "later_today") if late_day else ("pause", "within_15_min")

    if predicted_state == "neutral":
        if low_sleep and late_day:
            return "rest", "tonight"
        if early_day and energy >= 3:
            return "light_planning", "within_15_min"
        return "pause", "later_today"

    return "pause", "within_15_min"


def build_supportive_message(predicted_state: str, predicted_intensity: int, action: str, when: str) -> str:
    intensity_words = {
        1: "very lightly",
        2: "a bit",
        3: "moderately",
        4: "quite strongly",
        5: "very strongly",
    }
    return (
        f"You seem {intensity_words.get(int(predicted_intensity), 'somewhat')} {predicted_state} right now. "
        f"Try {action.replace('_', ' ')} {when.replace('_', ' ')}."
    )


def cross_validated_state_predictions(df: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    records: List[pd.DataFrame] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(df, df[STATE_COL]), start=1):
        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]

        model = build_state_pipeline()
        model.fit(train_fold, train_fold[STATE_COL])
        pred, conf = svm_confidence_from_margins(model, valid_fold)

        fold_frame = valid_fold[["id", TEXT_COL, STATE_COL]].copy()
        fold_frame["_row_id"] = valid_fold["_row_id"].to_numpy()
        fold_frame["fold"] = fold_id
        fold_frame["state_pred"] = pred
        fold_frame["state_confidence"] = conf
        records.append(fold_frame)

    return pd.concat(records, ignore_index=True).sort_values("_row_id").reset_index(drop=True)


def cross_validated_intensity_predictions(df: pd.DataFrame) -> pd.DataFrame:
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    records: List[pd.DataFrame] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(df), start=1):
        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]

        model = build_intensity_pipeline()
        model.fit(train_fold, train_fold[INTENSITY_COL])
        raw_pred = model.predict(valid_fold)
        rounded_pred = np.clip(np.rint(raw_pred), 1, 5).astype(int)
        conf = intensity_confidence_from_raw(raw_pred)

        fold_frame = valid_fold[["id", TEXT_COL, INTENSITY_COL]].copy()
        fold_frame["_row_id"] = valid_fold["_row_id"].to_numpy()
        fold_frame["fold"] = fold_id
        fold_frame["intensity_raw_pred"] = raw_pred
        fold_frame["intensity_pred"] = rounded_pred
        fold_frame["intensity_confidence"] = conf
        records.append(fold_frame)

    return pd.concat(records, ignore_index=True).sort_values("_row_id").reset_index(drop=True)


def benchmark_state_models(df: pd.DataFrame) -> Dict[str, float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        "text_only_svm": build_state_pipeline(),
        "text_plus_metadata": build_state_ablation_with_metadata(),
        "metadata_only": build_state_metadata_only(),
    }

    scores: Dict[str, List[float]] = {key: [] for key in models}
    for train_idx, valid_idx in splitter.split(df, df[STATE_COL]):
        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]
        for name, model in models.items():
            model.fit(train_fold, train_fold[STATE_COL])
            predictions = model.predict(valid_fold)
            scores[name].append(accuracy_score(valid_fold[STATE_COL], predictions))

    return {name: float(np.mean(values)) for name, values in scores.items()}


def benchmark_intensity_models(df: pd.DataFrame) -> Dict[str, float]:
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        "text_plus_metadata_mae": build_intensity_pipeline(),
        "metadata_only_mae": build_intensity_metadata_only(),
    }

    scores: Dict[str, List[float]] = {key: [] for key in models}
    for train_idx, valid_idx in splitter.split(df):
        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]
        for name, model in models.items():
            model.fit(train_fold, train_fold[INTENSITY_COL])
            raw_predictions = model.predict(valid_fold)
            rounded_predictions = np.clip(np.rint(raw_predictions), 1, 5)
            scores[name].append(mean_absolute_error(valid_fold[INTENSITY_COL], rounded_predictions))

    return {name: float(np.mean(values)) for name, values in scores.items()}


def state_feature_importance_table(model: Pipeline, top_n: int = 8) -> pd.DataFrame:
    classifier: LinearSVC = model.named_steps["classifier"]
    vectorizer: FeatureUnion = model.named_steps["vectorizer"]

    feature_names: List[str] = []
    for _, transformer in vectorizer.transformer_list:
        feature_names.extend(transformer.get_feature_names_out())
    feature_names = np.asarray(feature_names)

    rows = []
    for class_name, class_weights in zip(classifier.classes_, classifier.coef_):
        top_idx = np.argsort(class_weights)[-top_n:][::-1]
        for rank, feature_idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "emotional_state": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "weight": float(class_weights[feature_idx]),
                }
            )
    return pd.DataFrame(rows)


def assemble_oof_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    enriched_df = add_manual_features(df)
    state_oof = cross_validated_state_predictions(enriched_df)
    intensity_oof = cross_validated_intensity_predictions(enriched_df)

    merged = (
        enriched_df.merge(
            state_oof[["_row_id", "state_pred", "state_confidence"]],
            on="_row_id",
            how="left",
        )
        .merge(
            intensity_oof[["_row_id", "intensity_raw_pred", "intensity_pred", "intensity_confidence"]],
            on="_row_id",
            how="left",
        )
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    merged["combined_confidence"] = combine_confidence(
        merged,
        merged["state_confidence"].to_numpy(),
        merged["intensity_confidence"].to_numpy(),
        merged["state_pred"],
        merged["intensity_pred"],
    )
    merged["uncertain_flag"] = is_uncertain(merged, merged["combined_confidence"].to_numpy())
    merged["state_correct"] = (merged[STATE_COL] == merged["state_pred"]).astype(int)
    merged["intensity_correct"] = (merged[INTENSITY_COL] == merged["intensity_pred"]).astype(int)
    merged["intensity_within_1"] = (
        (merged[INTENSITY_COL] - merged["intensity_pred"]).abs() <= 1
    ).astype(int)

    state_metrics = {
        "state_accuracy": float(accuracy_score(merged[STATE_COL], merged["state_pred"])),
        "state_uncertainty_rate": float(merged["uncertain_flag"].mean()),
        "state_error_rate_when_uncertain": float(
            1.0
            - merged.loc[merged["uncertain_flag"] == 1, "state_correct"].mean()
            if (merged["uncertain_flag"] == 1).any()
            else 0.0
        ),
    }
    intensity_metrics = {
        "intensity_exact_accuracy": float(accuracy_score(merged[INTENSITY_COL], merged["intensity_pred"])),
        "intensity_mae": float(mean_absolute_error(merged[INTENSITY_COL], merged["intensity_pred"])),
        "intensity_within_1_accuracy": float(merged["intensity_within_1"].mean()),
    }
    return merged, state_metrics, intensity_metrics


def fit_final_models(train_df: pd.DataFrame):
    enriched_train = add_manual_features(train_df)
    state_model = build_state_pipeline()
    intensity_model = build_intensity_pipeline()
    state_model.fit(enriched_train, enriched_train[STATE_COL])
    intensity_model.fit(enriched_train, enriched_train[INTENSITY_COL])
    return state_model, intensity_model


def predict_submission(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    enriched_test = add_manual_features(test_df)
    state_model, intensity_model = fit_final_models(train_df)

    state_pred, state_conf = svm_confidence_from_margins(state_model, enriched_test)
    intensity_raw = intensity_model.predict(enriched_test)
    intensity_pred = np.clip(np.rint(intensity_raw), 1, 5).astype(int)
    intensity_conf = intensity_confidence_from_raw(intensity_raw)

    combined_conf = combine_confidence(
        enriched_test,
        state_conf,
        intensity_conf,
        state_pred,
        intensity_pred,
    )
    uncertain_flag = is_uncertain(enriched_test, combined_conf)

    output = test_df[["id"]].copy()
    output["predicted_state"] = state_pred
    output["predicted_intensity"] = intensity_pred
    output["confidence"] = combined_conf.round(4)
    output["uncertain_flag"] = uncertain_flag.astype(int)

    actions: List[str] = []
    times: List[str] = []
    messages: List[str] = []
    for row, state_label, intensity_value in zip(
        enriched_test.to_dict("records"),
        state_pred,
        intensity_pred,
    ):
        action, when = decide_action(pd.Series(row), state_label, int(intensity_value))
        actions.append(action)
        times.append(when)
        messages.append(build_supportive_message(state_label, int(intensity_value), action, when))

    output["what_to_do"] = actions
    output["when_to_do"] = times
    output["supportive_message"] = messages
    return output


def failure_case_table(oof_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    failures = oof_df[
        (oof_df["state_correct"] == 0) | (oof_df["intensity_correct"] == 0)
    ].copy()
    failures["short_text"] = failures["text_word_count"] <= 3
    failures["missing_metadata"] = failures[CAT_COLS].isna().sum(axis=1) > 0
    failures["conflicting_signals"] = (
        ((failures["stress_level"] >= 4) & failures["state_pred"].isin(["calm", "focused"]))
        | ((failures["energy_level"] <= 2) & (failures["state_pred"] == "focused"))
        | ((failures["reflection_quality"] == "conflicted") & failures["state_pred"].isin(["calm", "focused"]))
    )

    def assign_reason(row: pd.Series) -> str:
        if row["short_text"]:
            return "very short or vague reflection"
        if row["missing_metadata"]:
            return "missing context signal"
        if row["conflicting_signals"]:
            return "conflicting text and metadata"
        if row["reflection_quality"] == "conflicted":
            return "mixed language with contradictory cues"
        if row["journal_text"] in {"it was fine", "okay session", "a little lighter"}:
            return "label noise on repeated generic text"
        return "ambiguous wording"

    failures["failure_reason"] = failures.apply(assign_reason, axis=1)
    failures["combined_error"] = (
        (1 - failures["state_correct"]) + (1 - failures["intensity_correct"])
    )
    failures = failures.sort_values(
        ["combined_error", "combined_confidence", "text_word_count"],
        ascending=[False, True, True],
    )
    columns = [
        "_row_id",
        "id",
        TEXT_COL,
        STATE_COL,
        "state_pred",
        INTENSITY_COL,
        "intensity_pred",
        "combined_confidence",
        "failure_reason",
        "stress_level",
        "energy_level",
        "time_of_day",
        "reflection_quality",
    ]
    return failures[columns].head(top_n).reset_index(drop=True)


@dataclass
class TrainingSummary:
    state_metrics: Dict[str, float]
    intensity_metrics: Dict[str, float]
    state_ablation: Dict[str, float]
    intensity_ablation: Dict[str, float]


def build_training_summary(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, TrainingSummary]:
    enriched_train = add_manual_features(train_df)
    oof_frame, state_metrics, intensity_metrics = assemble_oof_frame(train_df)
    summary = TrainingSummary(
        state_metrics=state_metrics,
        intensity_metrics=intensity_metrics,
        state_ablation=benchmark_state_models(enriched_train),
        intensity_ablation=benchmark_intensity_models(enriched_train),
    )
    return oof_frame, summary


def save_predictions_csv(path: Path = Path("predictions.csv")) -> pd.DataFrame:
    train_df, test_df = load_datasets()
    predictions = predict_submission(train_df, test_df)
    predictions[
        [
            "id",
            "predicted_state",
            "predicted_intensity",
            "confidence",
            "uncertain_flag",
            "what_to_do",
            "when_to_do",
        ]
    ].to_csv(path, index=False)
    return predictions


if __name__ == "__main__":
    save_predictions_csv()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path("/home/saslab01/Desktop/replay_pad")

SUBGROUP_META_PATH = ROOT / "outputs" / "analysis" / "test_subgroup_metadata.csv"

MODEL_FILES = {
    "1-frame": ROOT / "outputs" / "predictions" / "image_baseline_1frame_test_video_predictions.csv",
    "5-frame average": ROOT / "outputs" / "predictions" / "image_baseline_5frame_test_video_predictions.csv",
    "10-frame average": ROOT / "outputs" / "predictions" / "image_baseline_10frame_test_video_predictions.csv",
    "CNN-LSTM": ROOT / "outputs" / "predictions" / "cnn_lstm_10frame_test_predictions.csv",
}

SAVE_CSV_PATH = ROOT / "outputs" / "analysis" / "experiment2_subgroup_metrics.csv"
SAVE_FIG_PATH = ROOT / "outputs" / "figures" / "experiment2_hand_fixed_acer.png"


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0.0
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return {
        "accuracy": acc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    subgroup_df = pd.read_csv(SUBGROUP_META_PATH).copy()
    rows = []

    for model_name, pred_path in MODEL_FILES.items():
        pred_df = pd.read_csv(pred_path).copy()

        required_cols = {"sample_id", "pred", "score_attack"}
        missing = required_cols - set(pred_df.columns)
        if missing:
            raise ValueError(f"{model_name} prediction file missing columns: {missing}")

        merged = pd.merge(
            subgroup_df,
            pred_df[["sample_id", "pred", "score_attack"]].copy(),
            on="sample_id",
            how="inner"
        )

        print(f"\n[INFO] {model_name}")
        print(f"[INFO] merged rows: {len(merged)}")

        # overall
        m = compute_metrics(merged["label"].values, merged["pred"].values)
        rows.append({
            "model": model_name,
            "subgroup_type": "overall",
            "subgroup_value": "all",
            **m
        })

        # attack_type
        for subgroup_value in ["real", "fixed", "hand"]:
            sub = merged[merged["attack_type"] == subgroup_value].copy()
            if len(sub) == 0:
                continue

            m = compute_metrics(sub["label"].values, sub["pred"].values)
            rows.append({
                "model": model_name,
                "subgroup_type": "attack_type",
                "subgroup_value": subgroup_value,
                **m
            })

        # environment
        for subgroup_value in ["controlled", "adverse"]:
            sub = merged[merged["environment"] == subgroup_value].copy()
            if len(sub) == 0:
                continue

            m = compute_metrics(sub["label"].values, sub["pred"].values)
            rows.append({
                "model": model_name,
                "subgroup_type": "environment",
                "subgroup_value": subgroup_value,
                **m
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"\n[INFO] saved -> {SAVE_CSV_PATH}")

    chart_df = result_df[
        (result_df["subgroup_type"] == "attack_type") &
        (result_df["subgroup_value"].isin(["fixed", "hand"]))
    ].copy()

    pivot_df = chart_df.pivot(index="model", columns="subgroup_value", values="acer")
    pivot_df = pivot_df.reindex(["1-frame", "5-frame average", "10-frame average", "CNN-LSTM"])

    ax = pivot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Experiment 2: ACER by hand vs fixed")
    ax.set_ylabel("ACER")
    ax.set_xlabel("Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=200)
    plt.close()

    print(f"[INFO] saved figure -> {SAVE_FIG_PATH}")


if __name__ == "__main__":
    main()
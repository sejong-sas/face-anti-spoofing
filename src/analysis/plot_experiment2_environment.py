from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/saslab01/Desktop/replay_pad")
SRC_PATH = ROOT / "outputs" / "analysis" / "experiment2_subgroup_metrics.csv"
SAVE_FIG_PATH = ROOT / "outputs" / "figures" / "experiment2_environment_acer.png"

def main():
    df = pd.read_csv(SRC_PATH)

    chart_df = df[
        (df["subgroup_type"] == "environment") &
        (df["subgroup_value"].isin(["controlled", "adverse"]))
    ].copy()

    pivot_df = chart_df.pivot(index="model", columns="subgroup_value", values="acer")
    pivot_df = pivot_df.reindex(["1-frame", "5-frame average", "10-frame average", "CNN-LSTM"])

    ax = pivot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Experiment 2: ACER by controlled vs adverse")
    ax.set_ylabel("ACER")
    ax.set_xlabel("Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=200)
    plt.close()

    print(f"[INFO] saved -> {SAVE_FIG_PATH}")

if __name__ == "__main__":
    main()
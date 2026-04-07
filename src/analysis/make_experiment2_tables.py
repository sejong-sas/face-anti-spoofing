from pathlib import Path
import pandas as pd

ROOT = Path("/home/saslab01/Desktop/replay_pad")
SRC_PATH = ROOT / "outputs" / "analysis" / "experiment2_subgroup_metrics.csv"

SAVE_ATTACK_PATH = ROOT / "outputs" / "analysis" / "experiment2_attacktype_table.csv"
SAVE_ENV_PATH = ROOT / "outputs" / "analysis" / "experiment2_environment_table.csv"

def main():
    df = pd.read_csv(SRC_PATH)

    attack_df = df[df["subgroup_type"] == "attack_type"].copy()
    attack_df = attack_df[attack_df["subgroup_value"].isin(["real", "fixed", "hand"])]
    attack_df = attack_df[["model", "subgroup_value", "accuracy", "apcer", "bpcer", "acer"]]
    attack_df.to_csv(SAVE_ATTACK_PATH, index=False)

    env_df = df[df["subgroup_type"] == "environment"].copy()
    env_df = env_df[env_df["subgroup_value"].isin(["controlled", "adverse"])]
    env_df = env_df[["model", "subgroup_value", "accuracy", "apcer", "bpcer", "acer"]]
    env_df.to_csv(SAVE_ENV_PATH, index=False)

    print(f"[INFO] saved -> {SAVE_ATTACK_PATH}")
    print(f"[INFO] saved -> {SAVE_ENV_PATH}")

if __name__ == "__main__":
    main()
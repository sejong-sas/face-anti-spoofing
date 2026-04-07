from pathlib import Path
import pandas as pd

ROOT = Path("/home/saslab01/Desktop/replay_pad")
META_PATH = ROOT / "metadata" / "metadata_all.csv"
SAVE_PATH = ROOT / "outputs" / "analysis" / "test_subgroup_metadata.csv"

def main():
    df = pd.read_csv(META_PATH)
    test_df = df[df["split"] == "test"].copy()

    test_df["sample_id"] = test_df["attack_type"].astype(str) + "/" + test_df["video_id"].astype(str)

    # 실험 2에서 바로 쓸 컬럼만 남김
    out_df = test_df[["sample_id", "video_id", "label", "attack_type", "video_path"]].copy()

    # 환경 조건 추출: controlled / adverse
    def parse_env(x):
        x = str(x).lower()
        if "controlled" in x:
            return "controlled"
        if "adverse" in x:
            return "adverse"
        return "unknown"

    out_df["environment"] = out_df["video_path"].apply(parse_env)

    out_df.to_csv(SAVE_PATH, index=False)
    print(f"[INFO] saved -> {SAVE_PATH}")
    print(f"[INFO] total test videos: {len(out_df)}")
    print("\n[INFO] attack_type counts")
    print(out_df["attack_type"].value_counts(dropna=False))
    print("\n[INFO] environment counts")
    print(out_df["environment"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
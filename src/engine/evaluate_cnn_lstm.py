import os
import json
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier
from src.evaluation.video_level_metrics import (
    search_best_threshold,
    apply_threshold_and_compute_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devel_csv",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip20_devel.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip20_test.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/outputs/checkpoints/cnn_lstm_clip20_random_best.pth",
    )
    parser.add_argument("--tag", type=str, default="cnn_lstm_clip20")
    parser.add_argument("--input_type", type=str, default="20-frame clip")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser.parse_args()


OUTPUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis", "devel_errors")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_loader(csv_path, split, img_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = ReplayPADClipDataset(
        csv_path=csv_path,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def aggregate_clip_to_video(clip_df, out_csv=None):
    agg_dict = {
        "label": "first",
        "split": "first",
        "attack_type": "first",
        "support_type": "first",
        "environment": "first",
        "dataset_name": "first",
        "score": "mean",
    }

    # client_id가 있을 때만 사용
    if "client_id" in clip_df.columns:
        agg_dict["client_id"] = "first"

    video_df = (
        clip_df.groupby("video_id", as_index=False)
        .agg(agg_dict)
    )

    if out_csv is not None:
        video_df.to_csv(out_csv, index=False)

    return video_df


def annotate_predictions(df, threshold):
    df = df.copy()
    df["threshold"] = float(threshold)
    df["pred_label"] = (df["score"] >= threshold).astype(int)
    df["correct"] = (df["pred_label"] == df["label"]).astype(int)

    def _error_type(row):
        if row["label"] == 0 and row["pred_label"] == 1:
            return "FP"
        elif row["label"] == 1 and row["pred_label"] == 0:
            return "FN"
        else:
            return "CORRECT"

    df["error_type"] = df.apply(_error_type, axis=1)
    return df


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[INFO] Saved JSON -> {path}")


def build_threshold_payload(args, threshold, devel_metrics):
    return {
        "model": "CNN-LSTM",
        "input_type": args.input_type,
        "threshold_selected_on": "devel",
        "threshold": round(float(threshold), 6),
        "devel_metrics": devel_metrics,
    }


def build_result_payload(args, threshold, devel_metrics, test_metrics, artifacts):
    return {
        "model": "CNN-LSTM",
        "input_type": args.input_type,
        "initialization": "random",
        "threshold_selected_on": "devel",
        "threshold": round(float(threshold), 6),
        "devel_threshold_search_result": devel_metrics,
        "test_video_metrics": test_metrics,
        "artifacts": artifacts,
    }


def inference_and_save_clip_predictions(model, csv_path, split, img_size, batch_size, num_workers, tag):
    _, loader = build_loader(csv_path, split, img_size, batch_size, num_workers)

    split_df = pd.read_csv(csv_path)
    split_df = split_df[split_df["split"] == split].reset_index(drop=True)

    all_scores = []
    model.eval()

    with torch.no_grad():
        for clips, _ in loader:
            clips = clips.to(DEVICE, non_blocking=True)
            logits = model(clips)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probs.detach().cpu().numpy().tolist())

    if len(all_scores) != len(split_df):
        raise ValueError(
            f"[ERROR] Number of scores ({len(all_scores)}) does not match split_df rows ({len(split_df)}) for split={split}"
        )

    split_df["score"] = all_scores

    out_csv = os.path.join(PRED_DIR, f"{tag}_{split}_clip_predictions.csv")
    split_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv} ({len(split_df)} rows)")

    pred_label_05 = (split_df["score"] >= 0.5).astype(int)
    clip_acc_05 = (pred_label_05 == split_df["label"]).mean()
    print(f"[INFO] {split} clip-level acc@0.5 = {clip_acc_05:.4f}")

    return split_df, out_csv


def save_misclassified_csv(video_df, split, tag):
    mis_df = video_df[video_df["error_type"].isin(["FP", "FN"])].copy()
    out_csv = os.path.join(ANALYSIS_DIR, f"{tag}_{split}_misclassified.csv")
    mis_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved {split} misclassified csv -> {out_csv} ({len(mis_df)} rows)")
    return mis_df, out_csv


def main():
    args = parse_args()
    os.makedirs(os.path.join(RESULT_DIR, args.tag), exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    model = CNNLSTMBinaryClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=DEVICE))
    print(f"[INFO] Loaded checkpoint: {args.checkpoint_path}")

    devel_clip_df, devel_clip_csv = inference_and_save_clip_predictions(
        model, args.devel_csv, "devel", args.img_size, args.batch_size, args.num_workers, args.tag
    )
    test_clip_df, test_clip_csv = inference_and_save_clip_predictions(
        model, args.test_csv, "test", args.img_size, args.batch_size, args.num_workers, args.tag
    )

    devel_video_csv = os.path.join(PRED_DIR, f"{args.tag}_devel_video_predictions.csv")
    test_video_csv = os.path.join(PRED_DIR, f"{args.tag}_test_video_predictions.csv")

    devel_video_df = aggregate_clip_to_video(devel_clip_df, devel_video_csv)
    test_video_df = aggregate_clip_to_video(test_clip_df, test_video_csv)

    print(f"[INFO] Devel videos: {len(devel_video_df)}")
    print(f"[INFO] Test videos: {len(test_video_df)}")

    best_threshold, devel_metrics = search_best_threshold(devel_video_df, step=0.001)
    test_metrics = apply_threshold_and_compute_metrics(test_video_df, best_threshold)

    print(f"[INFO] Best threshold selected on devel: {best_threshold:.6f}")

    devel_video_df = annotate_predictions(devel_video_df, best_threshold)
    test_video_df = annotate_predictions(test_video_df, best_threshold)

    devel_video_pred_csv = os.path.join(PRED_DIR, f"{args.tag}_devel_video_predictions_annotated.csv")
    test_video_pred_csv = os.path.join(PRED_DIR, f"{args.tag}_test_video_predictions_annotated.csv")

    devel_video_df.to_csv(devel_video_pred_csv, index=False)
    test_video_df.to_csv(test_video_pred_csv, index=False)

    devel_mis_df, devel_mis_csv = save_misclassified_csv(devel_video_df, "devel", args.tag)
    _, test_mis_csv = save_misclassified_csv(test_video_df, "test", args.tag)

    threshold_json = os.path.join(RESULT_DIR, args.tag, "devel_threshold.json")
    save_json(
        threshold_json,
        build_threshold_payload(args, best_threshold, devel_metrics),
    )

    result = build_result_payload(
        args=args,
        threshold=best_threshold,
        devel_metrics=devel_metrics,
        test_metrics=test_metrics,
        artifacts={
            "devel_clip_predictions_csv": devel_clip_csv,
            "test_clip_predictions_csv": test_clip_csv,
            "devel_video_predictions_csv": devel_video_csv,
            "test_video_predictions_csv": test_video_csv,
            "devel_video_predictions_annotated_csv": devel_video_pred_csv,
            "test_video_predictions_annotated_csv": test_video_pred_csv,
            "devel_misclassified_csv": devel_mis_csv,
            "test_misclassified_csv": test_mis_csv,
            "devel_threshold_json": threshold_json,
        },
    )

    out_json = os.path.join(RESULT_DIR, f"{args.tag}_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("[INFO] Final result")
    print(json.dumps(result, indent=2))
    print()
    print(f"[INFO] Saved result JSON -> {out_json}")

    fp_count = (devel_mis_df["error_type"] == "FP").sum()
    fn_count = (devel_mis_df["error_type"] == "FN").sum()
    print()
    print("[INFO] Devel misclassification summary")
    print(f"[INFO] FP count: {fp_count}")
    print(f"[INFO] FN count: {fn_count}")
    print(f"[INFO] Total misclassified: {len(devel_mis_df)}")


if __name__ == "__main__":
    main()

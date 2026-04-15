import os
import json
import ast
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class ReplayPADClipDataset(Dataset):
    def __init__(self, csv_path, split=None, transform=None):
        self.csv_path = csv_path
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"[ERROR] No rows found in dataset: csv_path={csv_path}, split={split}")

    def __len__(self):
        return len(self.df)

    def _parse_frame_paths(self, raw_value):
        # 이미 list인 경우
        if isinstance(raw_value, list):
            return raw_value

        # 문자열인 경우: JSON 또는 Python literal list 둘 다 대응
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(raw_value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            if "|" in raw_value:
                return [p for p in raw_value.split("|") if p]

        raise ValueError(f"[ERROR] Failed to parse frame_paths: {raw_value}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frame_paths = self._parse_frame_paths(row["frame_paths"])
        label = int(row["label"])

        frames = []
        for p in frame_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"[ERROR] Frame file not found: {p}")

            img = Image.open(p).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        # [T, C, H, W]
        clip = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return clip, label

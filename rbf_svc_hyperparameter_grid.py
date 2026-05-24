import argparse

import random

import numpy as np

from tqdm import tqdm

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models

from datasets import load_dataset

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split

# =========================

# 1. Args

# =========================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="Product", type=str)

    parser.add_argument("--target", default="Real World", type=str)

    parser.add_argument("--max-source", default=30000, type=int)

    parser.add_argument("--max-target", default=30000, type=int)

    parser.add_argument("--batch-size", default=64, type=int)

    parser.add_argument("--seed", default=42, type=int)

    # Source validation split

    parser.add_argument("--val-ratio", default=0.2, type=float)

    # RBF-SVC grid

    parser.add_argument("--c-grid", default="0.1,1,10,100", type=str)

    parser.add_argument("--gamma-grid", default="scale,auto,0.0001,0.001,0.01,0.1", type=str)

    return parser.parse_args()

# =========================

# 2. Utility

# =========================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")

def parse_grid_values(grid_str):

    values = []

    for item in grid_str.split(","):

        item = item.strip()

        if item in ["scale", "auto"]:

            values.append(item)

        else:

            values.append(float(item))

    return values

# =========================

# 3. Dataset

# =========================

class OfficeHomeDataset(Dataset):

    def __init__(self, hf_data, indices, transform):

        self.data = hf_data

        self.indices = indices

        self.transform = transform

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        sample = self.data[real_idx]

        img = sample["image"]

        if img.mode != "RGB":

            img = img.convert("RGB")

        x = self.transform(img)

        y = sample["label"]

        return x, y

def load_officehome(source, target, max_source, max_target, seed):

    dataset = load_dataset("flwrlabs/office-home")

    data = dataset["train"]

    source_idx = [i for i, d in enumerate(data["domain"]) if d == source]

    target_idx = [i for i, d in enumerate(data["domain"]) if d == target]

    rng = np.random.default_rng(seed)

    rng.shuffle(source_idx)

    rng.shuffle(target_idx)

    source_idx = source_idx[:max_source]

    target_idx = target_idx[:max_target]

    num_classes = len(set(data["label"]))

    print("Source domain:", source, "size:", len(source_idx))

    print("Target domain:", target, "size:", len(target_idx))

    print("Number of classes:", num_classes)

    return data, source_idx, target_idx, num_classes

# =========================

# 4. AlexNet Feature Extractor

# =========================

class AlexNetFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        weights = models.AlexNet_Weights.DEFAULT

        base = models.alexnet(weights=weights)

        self.transform = weights.transforms()

        self.features = base.features

        self.avgpool = base.avgpool

        # fc6, fc7까지만 사용하고 ImageNet classifier(fc8)는 제거한다.

        self.fc6_fc7 = nn.Sequential(

            base.classifier[0],

            base.classifier[1],

            base.classifier[2],

            base.classifier[3],

            base.classifier[4],

            base.classifier[5],

        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        feat = self.fc6_fc7(x)

        return feat

def build_feature_extractor(device):

    model = AlexNetFeatureExtractor()

    model = model.to(device)

    model.eval()

    for p in model.parameters():

        p.requires_grad = False

    return model, model.transform

# =========================

# 5. Feature Extraction

# =========================

def extract_features(model, loader, device, name="Extract"):

    model.eval()

    features = []

    labels = []

    with torch.no_grad():

        for x, y in tqdm(loader, desc=name):

            x = x.to(device)

            feat = model(x)

            features.append(feat.cpu().numpy())

            labels.append(y.numpy())

    features = np.concatenate(features, axis=0)

    labels = np.concatenate(labels, axis=0)

    return features, labels

# =========================

# 6. Evaluation and Grid Search

# =========================

def evaluate_prediction(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return acc, macro_f1

def svc_grid_search(x_train, y_train, x_val, y_val, c_grid, gamma_grid):

    print("\n==============================")

    print("Grid Search: Raw AlexNet Feature + RBF-SVC")

    print("==============================")

    best = {

        "c": None,

        "gamma": None,

        "val_acc": -1.0,

        "val_macro_f1": -1.0,

    }

    all_results = []

    for c in c_grid:

        for gamma in gamma_grid:

            clf = SVC(

                C=c,

                kernel="rbf",

                gamma=gamma,

            )

            clf.fit(x_train, y_train)

            pred = clf.predict(x_val)

            acc, macro_f1 = evaluate_prediction(y_val, pred)

            result = {

                "c": c,

                "gamma": gamma,

                "val_acc": acc,

                "val_macro_f1": macro_f1,

            }

            all_results.append(result)

            print(

                f"C={c}, gamma={gamma} | "

                f"Val Acc={acc:.4f}, Val Macro-F1={macro_f1:.4f}"

            )

            # Macro-F1 기준 선택, 동률이면 Accuracy 기준

            if (macro_f1 > best["val_macro_f1"]) or (

                macro_f1 == best["val_macro_f1"] and acc > best["val_acc"]

            ):

                best["c"] = c

                best["gamma"] = gamma

                best["val_acc"] = acc

                best["val_macro_f1"] = macro_f1

    print("\n==============================")

    print("Best Hyperparameter")

    print("==============================")

    print(f"C: {best['c']}")

    print(f"gamma: {best['gamma']}")

    print(f"Validation Accuracy: {best['val_acc']:.4f}")

    print(f"Validation Macro-F1: {best['val_macro_f1']:.4f}")

    print("\n==============================")

    print("Top-5 Results by Validation Macro-F1")

    print("==============================")

    all_results = sorted(

        all_results,

        key=lambda x: (x["val_macro_f1"], x["val_acc"]),

        reverse=True,

    )

    for rank, r in enumerate(all_results[:5], start=1):

        print(

            f"{rank}. C={r['c']}, gamma={r['gamma']} | "

            f"Val Acc={r['val_acc']:.4f}, "

            f"Val Macro-F1={r['val_macro_f1']:.4f}"

        )

    return best, all_results

# =========================

# 7. Main

# =========================

def main():

    args = parse_args()

    set_seed(args.seed)

    c_grid = parse_grid_values(args.c_grid)

    gamma_grid = parse_grid_values(args.gamma_grid)

    device = get_device()

    print("Device:", device)

    data, source_idx, target_idx, _ = load_officehome(

        args.source,

        args.target,

        args.max_source,

        args.max_target,

        args.seed,

    )

    feature_model, transform = build_feature_extractor(device)

    source_dataset = OfficeHomeDataset(data, source_idx, transform)

    target_dataset = OfficeHomeDataset(data, target_idx, transform)

    source_loader = DataLoader(

        source_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

    )

    target_loader = DataLoader(

        target_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

    )

    print("\n==============================")

    print("Feature Extraction with AlexNet fc7")

    print("==============================")

    xs, ys = extract_features(feature_model, source_loader, device, "Source Feature")

    # target feature는 hyperparameter tuning에 사용하지 않는다.

    # 단, 데이터 크기 확인을 위해 feature extraction만 수행한다.

    xt, yt = extract_features(feature_model, target_loader, device, "Target Feature")

    print("Source feature shape:", xs.shape)

    print("Target feature shape:", xt.shape)

    # =========================

    # Source train / validation split

    # =========================

    src_train_idx, src_val_idx = train_test_split(

        np.arange(len(xs)),

        test_size=args.val_ratio,

        random_state=args.seed,

        stratify=ys,

    )

    xs_train = xs[src_train_idx]

    ys_train = ys[src_train_idx]

    xs_val = xs[src_val_idx]

    ys_val = ys[src_val_idx]

    print("\nSource train size:", len(xs_train))

    print("Source validation size:", len(xs_val))

    # =========================

    # Scaling

    # =========================

    scaler = StandardScaler()

    xs_train_scaled = scaler.fit_transform(xs_train)

    xs_val_scaled = scaler.transform(xs_val)

    # =========================

    # Grid Search

    # =========================

    svc_grid_search(

        xs_train_scaled,

        ys_train,

        xs_val_scaled,

        ys_val,

        c_grid,

        gamma_grid,

    )

if __name__ == "__main__":

    main()

"""

Example:

python rbf_svc_hyperparameter_grid.py \

  --source Product \

  --target "Real World" \

  --max-source 30000 \

  --max-target 30000 \

  --batch-size 64 \

  --val-ratio 0.2 \

  --c-grid 0.1,1,10,100 \

  --gamma-grid scale,auto,0.0001,0.001,0.01,0.1

"""
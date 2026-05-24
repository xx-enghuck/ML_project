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

    # Fixed RBF-SVC hyperparameters

    parser.add_argument("--svc-c", default=10.0, type=float)

    parser.add_argument("--svc-gamma", default="0.0001", type=str)

    # CORAL numerical stability

    parser.add_argument("--coral-eps", default=1e-5, type=float)

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

def parse_gamma(gamma):

    if gamma in ["scale", "auto"]:

        return gamma

    return float(gamma)

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

        # AlexNet classifier:

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

# 6. CORAL Alignment

# =========================

def compute_covariance(x):

    n = x.shape[0]

    x_centered = x - np.mean(x, axis=0, keepdims=True)

    cov = (x_centered.T @ x_centered) / (n - 1)

    return cov

def coral_align(source_features, target_features, eps=1e-5):

    """

    Classical CORAL:

    Source feature를 target covariance structure에 맞게 변환한다.

    Xs_aligned = (Xs - mean_s) Cs^{-1/2} Ct^{1/2} + mean_t

    """

    xs = source_features.astype(np.float64)

    xt = target_features.astype(np.float64)

    mean_s = np.mean(xs, axis=0, keepdims=True)

    mean_t = np.mean(xt, axis=0, keepdims=True)

    xs_centered = xs - mean_s

    cov_s = compute_covariance(xs) + eps * np.eye(xs.shape[1])

    cov_t = compute_covariance(xt) + eps * np.eye(xt.shape[1])

    eig_s, vec_s = np.linalg.eigh(cov_s)

    eig_t, vec_t = np.linalg.eigh(cov_t)

    eig_s = np.maximum(eig_s, eps)

    eig_t = np.maximum(eig_t, eps)

    cov_s_inv_sqrt = vec_s @ np.diag(1.0 / np.sqrt(eig_s)) @ vec_s.T

    cov_t_sqrt = vec_t @ np.diag(np.sqrt(eig_t)) @ vec_t.T

    xs_aligned = xs_centered @ cov_s_inv_sqrt @ cov_t_sqrt + mean_t

    return xs_aligned.astype(np.float32)

# =========================

# 7. Evaluation

# =========================

def evaluate_svc(name, x_train, y_train, x_test, y_test, c, gamma):

    print("\n==============================")

    print(name)

    print("==============================")

    clf = SVC(

        C=c,

        kernel="rbf",

        gamma=gamma,

    )

    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)

    acc = accuracy_score(y_test, pred)

    macro_f1 = f1_score(y_test, pred, average="macro")

    print(f"SVC C: {c}")

    print(f"SVC gamma: {gamma}")

    print(f"Accuracy: {acc:.4f}")

    print(f"Macro-F1:  {macro_f1:.4f}")

    return {

        "name": name,

        "accuracy": acc,

        "macro_f1": macro_f1,

    }

# =========================

# 8. Main

# =========================

def main():

    args = parse_args()

    set_seed(args.seed)

    device = get_device()

    print("Device:", device)

    svc_gamma = parse_gamma(args.svc_gamma)

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

    xt, yt = extract_features(feature_model, target_loader, device, "Target Feature")

    print("Source feature shape:", xs.shape)

    print("Target feature shape:", xt.shape)

    # =========================

    # Raw baseline

    # =========================

    # SVC는 feature scale에 민감하므로 scaling을 적용한다.

    # Scaler는 source feature에 fit하고 target feature에는 transform만 적용한다.

    scaler_raw = StandardScaler()

    xs_raw = scaler_raw.fit_transform(xs)

    xt_raw = scaler_raw.transform(xt)

    raw_result = evaluate_svc(

        "Raw AlexNet Feature + RBF-SVC",

        xs_raw,

        ys,

        xt_raw,

        yt,

        args.svc_c,

        svc_gamma,

    )

    # =========================

    # CORAL alignment

    # =========================

    # Target label은 사용하지 않고 target feature만 covariance 계산에 사용한다.

    print("\n==============================")

    print("CORAL Alignment")

    print("==============================")

    xs_coral = coral_align(xs, xt, eps=args.coral_eps)

    scaler_coral = StandardScaler()

    xs_coral_scaled = scaler_coral.fit_transform(xs_coral)

    xt_coral_scaled = scaler_coral.transform(xt)

    coral_result = evaluate_svc(

        "CORAL-Aligned Feature + RBF-SVC",

        xs_coral_scaled,

        ys,

        xt_coral_scaled,

        yt,

        args.svc_c,

        svc_gamma,

    )

    # =========================

    # Final comparison

    # =========================

    print("\n==============================")

    print("Final Comparison")

    print("==============================")

    print("Raw AlexNet Feature + RBF-SVC")

    print(f"  Accuracy: {raw_result['accuracy']:.4f}")

    print(f"  Macro-F1:  {raw_result['macro_f1']:.4f}")

    print("CORAL-Aligned Feature + RBF-SVC")

    print(f"  Accuracy: {coral_result['accuracy']:.4f}")

    print(f"  Macro-F1:  {coral_result['macro_f1']:.4f}")

    print("\nImprovement")

    print(f"  Accuracy Diff: {coral_result['accuracy'] - raw_result['accuracy']:+.4f}")

    print(f"  Macro-F1 Diff:  {coral_result['macro_f1'] - raw_result['macro_f1']:+.4f}")

if __name__ == "__main__":

    main()

"""

Example:

python rbf_svc.py \

  --source Product \

  --target "Real World" \

  --max-source 30000 \

  --max-target 30000 \

  --batch-size 64 \

  --svc-c 10.0 \

  --svc-gamma 0.0001

"""
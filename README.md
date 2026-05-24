# ML_project

Office-Home 데이터셋을 사용해 **도메인 적응(Domain Adaptation)** 실험을 수행하는 머신러닝 프로젝트입니다.

이미지 데이터에서 AlexNet의 `fc7` feature를 추출한 뒤,  
KNN과 RBF-SVC 분류기를 사용해 source domain에서 학습하고 target domain에서 성능을 평가합니다.  
또한 CORAL(Correlation Alignment)을 적용하여 source feature를 target domain 분포에 맞춘 후 성능 변화를 비교합니다.

## 프로젝트 개요

이 프로젝트의 목적은 서로 다른 도메인 간 이미지 분류 성능 차이를 확인하고,  
CORAL 기반 feature alignment가 target domain 성능에 어떤 영향을 주는지 비교하는 것입니다.

기본 실험 설정은 다음과 같습니다.

```text
Source Domain: Product
Target Domain: Real World
Feature Extractor: AlexNet fc7
Classifiers: KNN, RBF-SVC
Evaluation Metrics: Accuracy, Macro-F1
```

## 폴더 구조

```text
ML_project/
├── download_office_home.py
├── knn.py
├── knn_hyperparameter_grid.py
├── rbf_svc.py
├── rbf_svc_hyperparameter_grid.py
├── .gitignore
└── README.md
```

## 주요 파일 설명

| 파일명 | 설명 |
|---|---|
| `download_office_home.py` | Office-Home 데이터셋을 다운로드하고 데이터 구조, 도메인, 클래스 수를 확인합니다. |
| `knn.py` | AlexNet feature를 추출한 뒤 KNN으로 raw feature와 CORAL-aligned feature 성능을 비교합니다. |
| `knn_hyperparameter_grid.py` | KNN의 `k`, distance metric, weight 옵션을 grid search로 탐색합니다. |
| `rbf_svc.py` | AlexNet feature를 추출한 뒤 RBF-SVC로 raw feature와 CORAL-aligned feature 성능을 비교합니다. |
| `rbf_svc_hyperparameter_grid.py` | RBF-SVC의 `C`, `gamma` 값을 grid search로 탐색합니다. |

## 사용 데이터셋

이 프로젝트는 Hugging Face의 Office-Home 데이터셋을 사용합니다.

```text
flwrlabs/office-home
```

Office-Home 데이터셋은 여러 시각 도메인으로 구성된 이미지 분류 데이터셋입니다.

사용 가능한 주요 도메인은 다음과 같습니다.

```text
Art
Clipart
Product
Real World
```

본 프로젝트에서는 기본적으로 `Product` 도메인을 source로, `Real World` 도메인을 target으로 사용합니다.

## 실험 방법

전체 실험 과정은 다음과 같습니다.

1. Office-Home 데이터셋 로드
2. Source domain과 target domain 분리
3. ImageNet pretrained AlexNet으로 이미지 feature 추출
4. Source feature를 이용해 분류기 학습
5. Target feature에서 성능 평가
6. CORAL alignment 적용
7. Raw feature와 CORAL-aligned feature 성능 비교

## CORAL Alignment

CORAL(Correlation Alignment)은 source domain feature의 covariance를 target domain feature의 covariance에 맞추는 비지도 도메인 적응 방법입니다.

본 프로젝트에서는 target label을 학습에 사용하지 않고, target feature의 통계 정보만 사용합니다.

개념적으로는 다음과 같은 변환을 수행합니다.

```text
Xs_aligned = (Xs - mean_s) Cs^(-1/2) Ct^(1/2) + mean_t
```

여기서 각 기호의 의미는 다음과 같습니다.

| 기호 | 의미 |
|---|---|
| `Xs` | Source feature |
| `mean_s` | Source feature 평균 |
| `mean_t` | Target feature 평균 |
| `Cs` | Source covariance matrix |
| `Ct` | Target covariance matrix |

## 설치 방법

필요한 Python 패키지는 다음과 같습니다.

```bash
pip install torch torchvision datasets scikit-learn numpy tqdm
```

Mac에서 Apple Silicon을 사용하는 경우 `mps`를 우선 사용합니다.  
CUDA가 가능한 환경에서는 `cuda`를 사용하고, 둘 다 사용할 수 없으면 `cpu`를 사용합니다.

## 실행 방법

### 1. 데이터셋 확인

```bash
python download_office_home.py
```

이 스크립트는 Office-Home 데이터셋을 다운로드하고, 데이터셋 구조와 도메인별 샘플 수를 출력합니다.

### 2. KNN 실험 실행

```bash
python knn.py
```

옵션을 직접 지정할 수도 있습니다.

```bash
python knn.py \
  --source Product \
  --target "Real World" \
  --max-source 30000 \
  --max-target 30000 \
  --batch-size 64 \
  --knn-neighbors 11 \
  --knn-metric cosine \
  --knn-weights distance
```

기본 KNN 설정은 다음과 같습니다.

```text
n_neighbors = 11
metric = cosine
weights = distance
```

### 3. KNN 하이퍼파라미터 탐색

```bash
python knn_hyperparameter_grid.py
```

기본 탐색 범위는 다음과 같습니다.

```text
k: 3, 5, 7, 9, 11
metric: cosine, euclidean, manhattan
weights: uniform, distance
```

직접 탐색 범위를 지정하려면 다음과 같이 실행합니다.

```bash
python knn_hyperparameter_grid.py \
  --k-grid 3,5,7,9,11 \
  --metric-grid cosine,euclidean,manhattan \
  --weight-grid uniform,distance
```

### 4. RBF-SVC 실험 실행

```bash
python rbf_svc.py
```

옵션을 직접 지정할 수도 있습니다.

```bash
python rbf_svc.py \
  --source Product \
  --target "Real World" \
  --max-source 30000 \
  --max-target 30000 \
  --batch-size 64 \
  --svc-c 10.0 \
  --svc-gamma 0.0001
```

기본 RBF-SVC 설정은 다음과 같습니다.

```text
C = 10.0
gamma = 0.0001
```

### 5. RBF-SVC 하이퍼파라미터 탐색

```bash
python rbf_svc_hyperparameter_grid.py
```

기본 탐색 범위는 다음과 같습니다.

```text
C: 0.1, 1, 10, 100
gamma: scale, auto, 0.0001, 0.001, 0.01, 0.1
```

직접 탐색 범위를 지정하려면 다음과 같이 실행합니다.

```bash
python rbf_svc_hyperparameter_grid.py \
  --c-grid 0.1,1,10,100 \
  --gamma-grid scale,auto,0.0001,0.001,0.01,0.1
```

## 평가 지표

본 프로젝트에서는 다음 두 가지 지표를 사용합니다.

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 예측 중 맞게 분류한 비율 |
| Macro-F1 | 클래스별 F1-score를 동일한 비중으로 평균낸 값 |

Macro-F1은 클래스 불균형이 있을 때 Accuracy보다 더 균형 잡힌 성능 비교가 가능합니다.

## 캐시 및 Git 관리

실험 과정에서 생성되는 feature cache 파일은 GitHub에 업로드하지 않습니다.

`.gitignore`에는 다음 항목을 포함하는 것을 권장합니다.

```text
.DS_Store
officehome_cache/
__pycache__/
*.pyc
```

이미 GitHub에 올라간 cache 파일을 제거하려면 다음 명령어를 실행합니다.

```bash
git rm --cached .DS_Store
git rm -r --cached officehome_cache
git add .gitignore
git commit -m "remove cache files from repository"
git push
```

위 명령어는 로컬 파일을 삭제하지 않고, Git 추적 대상에서만 제거합니다.

## 결과 해석

각 실험 스크립트는 raw AlexNet feature를 사용한 결과와 CORAL-aligned feature를 사용한 결과를 출력합니다.

출력 예시는 다음과 같은 형태입니다.

```text
Raw AlexNet Feature + KNN
  Accuracy: ...
  Macro-F1:  ...

CORAL-Aligned Feature + KNN
  Accuracy: ...
  Macro-F1:  ...

Improvement
  Accuracy Diff: ...
  Macro-F1 Diff:  ...
```

Accuracy Diff와 Macro-F1 Diff가 양수이면 CORAL 적용 후 target domain 성능이 향상된 것입니다.

## Author

xx-enghuck

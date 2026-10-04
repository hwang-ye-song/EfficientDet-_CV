# EfficientDet 실습 — 실습 코드 정리와 코드 리뷰

EfficientDet-D0([zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch))로 **사전학습 모델 추론 → 영상 분석 → 차량 데이터 학습·평가**를 진행하고, 실습 코드를 리뷰한 레포입니다.
과제 원본 노트북(Colab)과 강의 내용을 바탕으로, 내 컴퓨터(Windows · RTX 5060)에서 처음부터 끝까지 실행했습니다.

## ✅ 최종 정리본

- 📓 [EfficientDet_실습_코드리뷰.ipynb](EfficientDet_실습_코드리뷰.ipynb) — 실행 결과 포함
- 📄 [EfficientDet_실습_코드리뷰.pdf](EfficientDet_실습_코드리뷰.pdf) — 표지 포함 49쪽

원본 코드를 **순서 그대로** 실행하며 코드마다 아래에 의미를 정리했고, 고칠 점이 있는 코드는 **바로 아래에 🔍 코드 리뷰**(25개)를 붙였습니다.
진행하면서 막혔던 부분과 해결, 개인 회고는 [`개인회고.md`](개인회고.md)에 따로 정리했습니다.

## 📁 파일 구성

| 파일 | 설명 |
|---|---|
| [`EfficientDet_실습_코드리뷰.ipynb`](EfficientDet_실습_코드리뷰.ipynb) / [`.pdf`](EfficientDet_실습_코드리뷰.pdf) | **최종 정리본** — 내 컴퓨터에서 실행, 코드 설명 + 코드 리뷰 |
| [`개인회고.md`](개인회고.md) | 진행하면서 막혔던 부분과 해결, 개인 회고 |
| [`4. EfficientDet 실습 [프로젝트].ipynb`](4.%20EfficientDet%20%E1%84%89%E1%85%B5%E1%86%AF%E1%84%89%E1%85%B3%E1%86%B8%20%5B%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%5D.ipynb) | 과제 원본 노트북(Colab) |
| `soccer.mp4` | 영상 분석 입력. 원본 `soccer.gif`(437MB)는 GitHub 용량 제한(100MB) 때문에 1280px mp4로 변환해 올림 |
| `normal.png` | 제조 현장 예시 사진 (PCB 보드) |

- 데이터: [Kaggle — Vehicle Detection Image Dataset](https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset) (Roboflow COCO 형식, 흑백 v9i · 컬러 v8i)

---

## 🧩 EfficientDet은 어떤 모델인가

### 1-stage와 2-stage

| | 2-stage | 1-stage |
|---|---|---|
| 방식 | ① 물체가 있을 만한 후보 영역을 먼저 뽑고 ② 후보마다 다시 분류·박스 보정 | 미리 깔아 둔 위치(앵커·격자)마다 클래스와 박스를 **한 번에** 예측 |
| 대표 모델 | R-CNN, Fast/Faster R-CNN | YOLO, SSD, RetinaNet, **EfficientDet** |
| 특징 | 정확하지만 느림 | 빠르고, 후처리(NMS)로 겹친 박스를 정리 |

**EfficientDet은 YOLO와 같은 1-stage 모델**이다. RetinaNet처럼 앵커 기반으로, 특징맵 위치마다 앵커 9개(비율 3 × 크기 3)를 깔고 각 앵커의 클래스와 박스 보정값을 한 번에 예측한다.
이번 실습의 **"2단계 학습"(head만 학습 → 전체 학습)** 은 *학습 방법*이고, 모델 구조의 2-stage와는 다른 이야기다.

### 구조와 YOLO와의 차이

```
입력 이미지 → EfficientNet (백본, 특징 추출) → BiFPN (여러 크기 특징 섞기) → class head + box head → NMS
```

| | EfficientDet | YOLOv8 (지난 실습) |
|---|---|---|
| 방식 | 1-stage, **앵커 기반** (위치마다 앵커 9개) | 1-stage, 앵커 없음(anchor-free) |
| 특징 섞기 | **BiFPN** — 위아래 방향으로 여러 번 섞고, 입력마다 가중치를 둠 | PAN-FPN |
| 크기 조절 | **Compound Scaling** — D0~D7로 입력 크기·깊이·너비를 함께 키움 | n / s / m / l / x |
| 사용 코드 | 레포의 함수를 단계별로 직접 호출 (전처리 → 추론 → 후처리 → 좌표 복원) | `model.predict()` 한 줄 |

| 모델 | 입력 크기 | 파라미터 | COCO mAP |
|---|---|---|---|
| D0 (이번 실습) | 512 | 3.9M | 33.8 |
| D1 | 640 | 6.6M | 39.6 |
| D3 | 896 | 12M | 47.5 |

---

## 🔄 실습 흐름과 결과

### Part 1. 사전학습 모델로 이미지 추론

| 단계 | 확인한 값 |
|---|---|
| 전처리 | 1920×1080 → 비율 유지 512×288 + 아래 패딩 224 → 512×512 (`framed_metas = (512, 288, 1920, 1080, 0, 224)`) |
| 모델 출력 | 특징맵 5개(64·32·16·8·4), 앵커 **49,104개** = (64²+32²+16²+8²+4²) × 9 |
| 후처리 | 점수 0.2 미만 제거 + NMS → **박스 37개** |
| 좌표 복원 | `invert_affine`으로 512 기준 좌표를 원본 크기로 |

### Part 2. 영상 분석
`soccer.gif`(2732×1440, 367프레임)를 프레임마다 같은 과정으로 탐지해 mp4로 합쳤다(RTX 5060에서 3~4분). 사람과 벤치는 잘 잡았지만, 축구공은 512로 줄이면 지름이 약 16픽셀이 되어 잡지 못했다.

### Part 3. 차량 데이터 학습·평가

| 데이터 분석 | 결과 |
|---|---|
| RGB 평균 / 표준편차 | `[0.464, 0.472, 0.470]` / `[0.165, 0.164, 0.163]` — 거의 회색조 |
| 박스 | 2,069개, 크기 중앙값 **17.7px**(작은 물체), 비율 중앙값 0.67(세로로 긴 박스) |
| 추천 앵커 | 비율 `[(0.52, 1.0), (0.67, 1.0), (0.82, 1.0)]`, 크기 `[0.54, 1.0, 2.13]` |

| 학습 (흑백 train 136장) | 검증 손실 변화 |
|---|---|
| 1단계: head만 10에폭 | Classification 13427 → 82, Regression 3.21 → 2.16 |
| 2단계: 전체 학습 (에폭 10~29) | Classification 17.2 → 0.74, Total 최저 3.03 (24에폭) |

| 평가 (valid, COCO 방식) | 내 컴퓨터 | 원본 Colab |
|---|---|---|
| AP @[0.50:0.95] | 0.005 | 0.000 |
| AP @0.50 | 0.011 | 0.000 |
| AR @100 (small / medium / large) | 0.027 / 0.198 / 0.331 | 0.002 / 0.097 / 0.104 |

학습은 진행됐지만 차를 거의 맞히지 못했다. 작은 차일수록 못 찾는데, 박스 크기 중앙값(17.7px)이 D0의 가장 작은 앵커(약 17px)와 비슷하거나 더 작고, 학습 데이터도 136장으로 적다. 원본 Colab에서는 2단계 학습이 중간에 끊겨 1단계 가중치로 평가했다.

---

## 💻 내 컴퓨터에서 실행하려고 바꾼 곳

원본은 Colab(리눅스)용이라, Windows에서 그대로 안 되는 줄만 바꿨다. 정리본에는 바꾼 줄마다 `# ☁️ 코랩용 (원본)`과 `# 💻 노트북용`을 나란히 적었다. 원본 코드 셀 57개 중 43개는 글자 하나까지 그대로다.

| 원본 (Colab) | 바꾼 코드 | 이유 |
|---|---|---|
| `!mkdir -p weights`, `!wget ...` | `!if not exist weights mkdir weights`, `!curl -L ...` | Windows 명령창의 `mkdir`은 `-p`를 모르고, `wget`이 없다 |
| `!pwd`, `!ls -l`, `!unzip ...`, `ls -Art \| grep` | `os.getcwd()`, `os.listdir`, `zipfile`, `dir /b /od` | Windows 명령창에는 리눅스 명령이 없다 |
| `soccer.mp4`, `/content/...` 경로 | `soccer.gif`, 레포 기준 상대 경로 | 받은 영상 파일은 gif이고, `/content/`는 Colab에만 있다 |
| `!pip install` | `%pip install` | `!pip`는 시스템 파이썬에 설치된다 |
| `! python train.py ...` | `!{sys.executable} train.py ... -n 0` | 시스템 파이썬에는 `pycocotools`가 없다. 워커 12개는 Windows에서 에폭당 3분 넘게 걸려 0으로 줄였다(→ 약 15초) |
| yml (마크다운 메모) | 파이썬으로 파일 생성, `project_name`·`num_gpus` 수정 | 원본은 파일을 직접 만들었고, 메모 값이 실제 폴더·GPU 설정과 달랐다 |

---

## 🧱 진행하면서 막혔던 부분과 해결

| 막힌 부분 | 원인 | 해결 |
|---|---|---|
| 원본 노트북을 내 컴퓨터에서 그대로 실행할 수 없음 | 원본은 Colab용이라 `/content/...` 경로, `wget`, `mkdir -p`가 들어 있다 | 안 되는 곳만 고치고 `☁️ 코랩용`과 `💻 노트북용`을 나란히 표시했다 |
| `!ls -l`, `!pwd`가 `'ls' is not recognized...` 에러 | Jupyter가 `!` 명령을 Windows 명령창(cmd)으로 실행해서 `ls`·`pwd`·`unzip`·`grep`이 없다. 처음엔 Git Bash에서 시험해 된다고 잘못 판단했다 | `os.getcwd()`, `os.listdir`, `zipfile`, `dir /b /od`로 바꿨다 |
| `soccer.mp4`가 없음 | 원본 Colab 실행 기록에도 `False`였다. 실제로 쓰인 파일은 `soccer.gif`였다(결과 프레임 크기 2732×1440이 같음) | `soccer.gif`를 받아 `video_src`를 바꿨다 |
| yml 메모대로 하면 데이터를 못 찾음 | `project_name`이 폴더 이름과 달랐다 | `my_car_detect_proj`로 맞췄다. 원본 Colab 로그도 이 이름의 폴더에 저장되어 있었다 |
| `num_gpus: 0`이면 CPU로 학습함 | `train.py`가 0이면 GPU를 끈다 | `num_gpus: 1`로 바꿨다 |
| 학습이 시작되지 않음 (`UnicodeDecodeError: 'cp949'`) | yml 파일 안에 한글·이모지 주석을 넣었더니 `train.py`가 yml을 cp949로 읽다가 실패했다. `!` 명령은 실패해도 셀이 멈추지 않아 다음 셀에서야 드러났다 | yml은 파이썬으로 쓰고 파일 안에는 원본 영어 주석만 남겼다 |
| `ReduceLROnPlateau(..., verbose=True)` 에러 | 최신 PyTorch에서 `verbose` 인자가 없어졌다 | 원본 안내대로 지웠다 |
| 학습이 에폭당 3분 30초 걸림 | 워커 12개를 Windows에서는 에폭마다 새로 띄운다. 그동안 GPU 사용률은 0%였다 | `-n 0`으로 에폭당 약 15초가 됐다 |
| `! python train.py` 실패 | `python`이 `pycocotools`가 없는 시스템 파이썬을 가리킨다 | `!{sys.executable}`로 커널의 파이썬을 썼다 |
| PCB 사진 추론에서 `'NoneType' object is not subscriptable` | 커널이 이미 레포 안에 있는 상태에서 `git clone`·`os.chdir`를 다시 실행해 레포가 두 겹으로 받아졌고, 안쪽 사본에는 이미지가 없었다 | 안쪽 사본을 지우고 커널을 재시작해 처음부터 다시 실행했다 |

---

## ⚙️ 실행 환경

| 항목 | 내용 |
|---|---|
| 내 컴퓨터 | Windows 11 · Python 3.11 · PyTorch 2.11 (CUDA 12.8) · RTX 5060 Laptop 8GB |
| 패키지 | `torch torchvision` (cu128), `pycocotools opencv-python matplotlib tqdm tensorboardX webcolors pyyaml` |
| 준비물 | 레포 clone, `weights/efficientdet-d0.pth`, `datasets/archive.zip`(Kaggle), `soccer.gif` |

RTX 50 시리즈는 CUDA 12.8 빌드가 필요해서 PyTorch를 아래처럼 설치했다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

# EfficientDet 실습 보고서

> EfficientDet-D0(PyTorch)으로 **사전학습 모델 추론 → 영상 추론 → 커스텀 차량 데이터 학습·평가**까지 진행한 실습 정리

## 📁 파일 구성

| 파일 | 설명 |
|---|---|
| [`EfficientDet_Colab.ipynb`](./EfficientDet_Colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hwang-ye-song/EfficientDet-_CV/blob/main/EfficientDet_Colab.ipynb) | **Colab 실행용** — 원본 코드는 그대로 두고, 문제가 생기는 셀 위에만 `🔧 [추가]` 셀을 넣은 버전 |
| [`4. EfficientDet 실습 [정리본].ipynb`](./4.%20EfficientDet%20실습%20[정리본].ipynb) | **정리본** — 원본 코드·출력 그대로 + 각 셀 아래 `💬 코멘트`, 결과 분석, 느낀 점 |
| [`4. EfficientDet 실습 [프로젝트].ipynb`](./4.%20EfficientDet%20실습%20[프로젝트].ipynb) | 원본 실습 노트북 (Colab 실행) |
| `4. EfficientDet 실습 [프로젝트].pdf` | 강의 자료 — EfficientDet 기반 제조 영상 분석 최적화 |
| `normal.png` | 제조 영상 예시 (PCB 보드) |
| `soccer.mp4` | 영상 추론 실습의 **입력 영상** (축구). 원본 `soccer.gif`(437MB)를 1280px mp4로 변환 — 아래 [작업 기록](#-작업-기록) 참고 |

- 사용 구현체: [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- 커스텀 데이터: [Kaggle — Vehicle Detection Image Dataset](https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset)

---

## 🔄 실습 흐름

```
1. 환경 설정       저장소 clone → D0 사전학습 가중치 → 모델 로드 (COCO 90클래스)
2. 이미지 추론     전처리(letterbox 512) → forward → 후처리(BBox 변환·Clip·NMS) → 좌표 복원 → 시각화
3. 영상 추론       프레임 단위로 2번 반복 → 프레임 저장 → mp4 합치기
4. 데이터 준비     COCO 라벨 확인 → mean/std 계산 → 앵커 분석 → 폴더·json 변환 → yml 작성
5. 학습            head만 10 epoch → 전체 fine-tuning (중단)
6. 평가·추론       COCO mAP 평가 → 테스트 이미지 추론
```

### 1~2. 사전학습 모델 추론

```python
model = EfficientDetBackbone(compound_coef=0, num_classes=90)   # D0, COCO 90클래스
model.load_state_dict(torch.load('weights/efficientdet-d0.pth'))
model.requires_grad_(False); model.eval()                        # 추론 모드

ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=512)  # 비율 유지 + 패딩
features, regression, classification, anchors = model(x)
out = postprocess(x, anchors, regression, classification,
                  BBoxTransform(), ClipBoxes(), threshold, iou_threshold)  # 박스 변환 + NMS
out = invert_affine(framed_metas, out)                           # 512 좌표 → 원본 좌표
```

| 확인한 값 | 의미 |
|---|---|
| 원본 `(1080, 1920)` → 입력 `(512, 512)` | 가로 기준 축소 후 세로 224px 패딩 |
| 특징맵 5개: 64→32→16→8→4, 채널 64 | P3~P7 다중 스케일 (BiFPN 출력) |
| 앵커 **49,104개** = (64²+32²+16²+8²+4²) × 9 | 모든 위치 × (비율 3 × 크기 3) |
| 최종 박스 **37개** | threshold + NMS로 걸러진 결과 |

### 3. 영상 추론
영상은 이미지의 연속이므로 `cv2.VideoCapture`로 프레임을 읽고 위 과정을 프레임마다 반복 → jpg로 저장 → `cv2.VideoWriter`로 20fps mp4 생성 (`output_soccer.mp4`). 입력 영상은 `soccer.mp4`.

### 4. 커스텀 데이터 준비 (차량 6클래스)

| 항목 | 결과 |
|---|---|
| RGB mean / std | `[0.464, 0.472, 0.470]` / `[0.165, 0.164, 0.163]` |
| 박스 수 | 2,069개 (Car 1527 · Motorcycle 281 · Pickup 190 · Truck 47 · Bus 24) |
| 박스 크기 중앙값 | **17.7px** → 매우 작은 물체 |
| 추천 앵커 | ratios `[(0.52,1.0),(0.67,1.0),(0.82,1.0)]`, scales `[0.54, 1.0, 2.13]` |
| obj_list | `['cars', 'Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']` |

### 5. 학습

```bash
# 1단계: backbone·BiFPN 고정, head만 학습
python train.py -c 0 -p my_car_detect_proj --head_only True --lr 1e-3 --batch_size 16 \
       --load_weights weights/efficientdet-d0.pth --num_epochs 10
# 2단계: 전체 fine-tuning (속도 문제로 첫 step에서 중단)
python train.py -c 0 -p my_car_detect_proj --head_only False ... --num_epochs 30
```

| Epoch | 0~1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Val Total loss | – | 5805 | 2687 | 1215 | 528 | 275 | 175 | 132 | **110** |

- head 크기가 COCO(90×9=810채널) → 커스텀(6×9=54채널)으로 바뀌어 그 층만 새로 초기화됨 (`size mismatch` 경고는 정상)

### 6. 평가

| 지표 | 값 |
|---|---|
| AP @[0.50:0.95] | **0.000** |
| AP @0.50 | 0.000 |
| AP (small / medium / large) | 0.000 / 0.002 / 0.068 |
| AR @100 | 0.042 |

---

## 📊 결과 분석 및 개선 방안

loss는 꾸준히 줄었지만 AP는 0. 코드를 따라가며 찾은 원인 후보:

| # | 문제 | 개선 방안 |
|---|---|---|
| 1 | **학습·추론 앵커 불일치** — 학습(yml) `[(0.52,1.0),(0.67,1.0),(0.82,1.0)]` vs 추론 코드 `[(1.0,1.0),(1.3,0.8),(1.9,0.5)]` | 추론에도 yml과 같은 ratios/scales 사용 |
| 2 | **분석·학습·테스트 데이터 버전이 다름** — 분석은 컬러(v8i), 학습은 흑백(v9i), 테스트는 컬러(v8i) | 한 가지 버전으로 통일 |
| 3 | **학습량 부족** — head만 10 epoch, 전체 학습은 중단. yml `num_gpus: 0`이면 CPU로 학습됨 | GPU 사용 후 충분히 학습 |
| 4 | **obj_list의 `'cars'`** — 저장소는 `category_id - 1`을 라벨로 사용 → 클래스 이름이 한 칸 밀릴 수 있음 | `'cars'` 제거 (5클래스) |
| 5 | **작은 물체 + 클래스 불균형** | 입력 해상도 키우기(D1~D2), 증강, 불균형 보정 |
| 6 | 변환한 json을 원본 복사 셀이 다시 덮어씀 | 하나만 실행 |

> classification loss가 수천 단위로 시작한 것도(보통 1 안팎) 앵커·라벨 설정이 데이터와 맞지 않았다는 신호로 보인다.

---

## ✍️ 느낀 점

**1. 사전학습 모델의 위력**
코드 몇 줄과 15MB짜리 가중치만으로 처음 보는 이미지와 축구 영상에서 사람·공을 바로 잡아내는 것이 인상적이었다. 4만 9천 개의 앵커 후보가 후처리를 거쳐 37개로 줄어드는 과정을 직접 출력해 보며 "탐지 모델 = 후보를 많이 깔고 걸러내는 구조"라는 것을 숫자로 이해했다.

**2. 데이터 준비가 학습보다 어렵다**
`train.py` 실행은 한 줄이었지만, 그 한 줄을 위해 폴더 구조, COCO json 변환, mean/std, 앵커 분석, yml 작성까지 훨씬 많은 준비가 필요했다. 경로나 데이터 버전 하나만 달라도 결과가 완전히 달라질 수 있다는 걸 체감했다.

**3. 학습 결과의 아쉬움 — 일관성의 중요성**
loss가 5805 → 110으로 줄어 잘 되는 줄 알았지만 AP는 0이었다. 학습과 추론의 앵커가 달랐고, 분석한 데이터와 학습한 데이터도 달랐다. **"loss 감소 ≠ 좋은 모델"**, 그리고 학습·추론·평가에서 설정을 똑같이 유지하는 것의 중요성을 배웠다. 다음에는 설정을 통일하고 GPU로 충분히 학습해 결과를 비교해 보고 싶다.

**4. 제조 현장 적용 가능성**
PCB 사진(`normal.png`)처럼 제조 현장 이미지에 적용하면 부품 누락·불량 위치 탐지에 쓸 수 있을 것 같다. PCB 부품·결함도 이번 차량처럼 **작은 물체**라서 앵커 분석과 입력 해상도 조절이 그대로 중요하고, D0처럼 가벼운 모델은 실시간 라인 검사에도 활용할 수 있을 것이다.

---

## 🛠 작업 기록

### 1. 실습 정리
- 원본 노트북의 코드·출력은 그대로 두고, 모든 코드 셀 아래에 `💬 코멘트`를 단 **정리본**을 만들었다.
- 결과(AP 0.000)의 원인을 코드에서 찾아 **결과 분석 및 개선 방안**으로 정리하고, 느낀 점을 덧붙였다.

### 2. 다시 실행했을 때 "이미지가 없다"는 에러가 난 원인
원본 노트북은 `git clone`으로 받을 수 없는 파일을 **직접 업로드한 상태**에서 실행한 것이라, 처음부터 다시 돌리면 파일을 찾지 못했다.

| 없던 파일 | 쓰는 곳 | 증상 |
|---|---|---|
| `soccer.mp4` | 영상 추론 | 프레임이 저장되지 않아 `frame_00004.jpg`를 찾지 못함 |
| `datasets/archive.zip` | 커스텀 데이터 | "이미지를 불러올 수 없습니다", 경로 에러 |
| `projects/my_car_detect_proj.yml` | 학습·평가 | yml은 화면 캡처로만 남아 있어서 파일이 없음 |

그 밖에 런타임이 초기화되면 파일이 지워지는 점, 셀을 다시 실행하면 경로가 꼬이는 점(`os.chdir`, `%cd`, `img_path` 덮어쓰기)도 원인이었다.

### 3. `soccer.gif`의 정체
처음에는 `soccer.gif`를 탐지 결과로 생각했지만, 프레임을 확인해 보니 **탐지 박스가 없는 입력 영상**이었다(`soccer.mp4`를 GIF로 바꿔 둔 것).
- 437MB라 GitHub에 올릴 수 없어서(파일당 100MB 제한) **1280px · 25fps H.264 mp4(5.6MB)로 변환**해 `soccer.mp4`로 올렸다.
- **원본과 해상도가 다른 것은 용량 제한 때문에 어쩔 수 없었다.** 다만 모델 입력은 어차피 512px로 줄어들기 때문에 탐지 결과에는 거의 영향이 없다.

### 4. Colab 실행용 노트북 (`EfficientDet_Colab.ipynb`)
처음에는 여러 셀을 한꺼번에 고쳤지만, **원본 흐름을 최대한 따라가는 것이 좋다**고 판단해서 방식을 바꿨다.
→ 원본 코드 셀 57개는 **내용·순서 그대로** 두고, 처음부터 실행할 때 막히는 셀 **바로 위에 `🔧 [추가]` 셀만** 넣었다.

| 막히는 원본 셀 | 원인 | 위에 추가한 것 |
|---|---|---|
| (노트북 맨 위 안내) | `soccer.mp4`가 없음 | 저장소의 `soccer.mp4`를 받아 `Yet-Another-EfficientDet-Pytorch` 폴더에 업로드하도록 안내 |
| `!unzip ./datasets/archive.zip ...` | `archive.zip`이 없음 | Kaggle에서 자동 다운로드 (로그인 불필요) |
| `train.py` 실행 | yml 파일이 캡처로만 있음 | `%%writefile`로 원본 yml 내용 그대로 생성 (`project_name`을 실제 폴더명으로, `num_gpus`를 1로만 변경) |
| `train.py` 실행 | 최신 PyTorch의 `verbose=True` 에러 | 원본 안내와 같은 수정을 `sed`로 자동 적용 |
| 2단계 학습 | `d0_9_80.pth` 파일명을 직접 지정 | 저장된 가중치 목록을 먼저 확인 |

- 빈 셀과 이미지 파일이 없는 캡처 셀만 뺐다.
- AP 0.000의 원인(앵커 불일치, 데이터 버전 혼용 등)은 원본 그대로 두었다. 분석과 개선 방안은 정리본에 있다.

**검증 범위**: 모든 코드 셀 문법 검사, 원본 코드 셀 57개 보존 확인, 새 환경 기준으로 데이터 다운로드 → 압축 해제 → 변환 → yml 생성까지 실제 실행 확인. 학습·평가·추론은 GPU 환경(Colab)에서 실행 필요.

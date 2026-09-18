# Roundabout-SSM

rounD 데이터셋 기반 SSM(Mamba2) 회전교차로 진입 판단 및 주행계획 예측 프로젝트

## 프로젝트 개요

회전교차로에 진입하는 차량의 과거 주행 정보와 주변 차량의 움직임을 이용하여
다음 항목을 예측하는 모델을 개발하였다.

- GO / WAIT 진입 판단
- Time-to-Entry
- 진입 속도
- 진입 각도
- 향후 4초 미래 주행 궤적

본 프로젝트에서는 LSTM과 Mamba2 기반 모델을 비교하고,
회전교차로 내부 차량과의 TTC 및 temporal gap 정보를 추가하여
최종 Mamba2 기반 예측 모델을 구성하였다.

---

## 1. 데이터셋 및 실험 대상

본 프로젝트에서는 실제 회전교차로 차량 궤적이 포함된 **rounD Dataset**을 사용하였다.

rounD Dataset은 차량별 위치, 속도, 가속도, 진행방향(heading) 등의 정보를
25 Hz 주기로 제공하며, 항공 영상 기반의 실제 교통 상황을 포함하고 있다.

전체 24개 recording 중 동일한 회전교차로를 공유하는 **Location 0의 recording 02~23**을
주 학습 및 평가 대상으로 사용하였다.

### 데이터 구성

| 항목 | 값 |
|---|---:|
| 사용 recording | 02 ~ 23 |
| Frame Rate | 25 Hz |
| 추출한 진입 이벤트 | 11,354개 |
| 전체 decision sample | 107,691개 |
| Train sample | 77,644개 |
| Validation sample | 16,751개 |
| Test sample | 13,296개 |

데이터 누수를 방지하기 위해 개별 sample을 무작위로 나누지 않고
**recording 단위로 Train / Validation / Test를 분리**하였다.

---

## 2. 입력 데이터 구성

각 decision sample은 차량이 회전교차로 진입선을 통과하기 전 시점에서 생성하였다.

모델은 현재 시점을 기준으로 **과거 2초 동안의 Ego 차량 및 주변 차량 움직임**을 입력으로 사용한다.

- History length: 2.0초
- Frame rate: 25 Hz
- History frames: 50
- Ego vehicle: 1대
- Neighbor vehicle: 최대 8대
- 총 Agent 수: 최대 9대
- 주변 차량 탐색 반경: 30 m

모든 차량 상태는 현재 Ego 차량을 기준으로 한 **Ego-centric 좌표계**로 변환하였다.

현재 Ego 차량은 다음과 같이 정규화된다.

```text
현재 Ego 위치    = (0, 0)
현재 Ego heading = 0°
전방             = +x
좌측             = +y
```

각 agent의 시계열 feature는 다음 9개로 구성하였다.

```text
rel_x
rel_y
rel_vx
rel_vy
rel_ax
rel_ay
sin(relative_heading)
cos(relative_heading)
presence
```

최종 시계열 입력 크기는 다음과 같다.

```text
[9 agents, 50 frames, 9 features]
```
### Ego-centric 입력 데이터 예시

아래 그림은 실제 학습 sample을 Ego-centric 좌표계로 변환한 결과이다.
현재 Ego 차량을 원점으로 두고, 과거 2초 동안의 Ego 궤적과 주변 차량 궤적,
회전교차로 진입선을 함께 구성하였다.

<p align="center">
  <img src="images/dataset_ego_centric_samples.png" width="900">
</p>

<p align="center">
  <em>Ego-centric 좌표계로 변환된 decision sample 예시</em>
</p>
---

## 4. Entry Line 정의 및 진입 이벤트 추출

rounD Dataset에는 차량의 연속적인 주행 궤적은 제공되지만,
차량이 정확히 어느 시점에 회전교차로에 진입했는지를 나타내는 label은 별도로 존재하지 않는다.

따라서 Location 0의 회전교차로에 대해 각 진입로별 **Entry Line**을 직접 정의하고,
차량 궤적이 해당 선을 통과하는 순간을 실제 진입 이벤트로 추출하였다.

### 4.1 Entry Line 정의

Location 0에는 총 4개의 진입로가 존재하며,
각 진입로에 Entry 0 ~ Entry 3을 시계방향으로 부여하였다.

<p align="center">
  <img src="images/location0_entry_lines.png" width="900">
</p>

<p align="center">
  <em>rounD Location 0의 4개 Entry Line 정의</em>
</p>

이미지 좌표로 지정한 Entry Line은 rounD Dataset의 `orthoPxToMeter` 정보를 이용하여
실제 world coordinate로 변환하였다.

Location 0의 recording 02~23은 동일한 회전교차로와 좌표계를 공유하기 때문에
한 번 정의한 Entry geometry를 모든 recording에 공통으로 적용할 수 있었다.

---

### 4.2 Entry Crossing 검출

각 차량의 연속된 두 frame 사이의 궤적 segment와 Entry Line의 교차 여부를 검사하여
실제 crossing frame을 검출하였다.

단순히 선과 교차하는 경우만 사용하는 것이 아니라,
차량이 회전교차로 중심 방향으로 이동하는 경우만 진입 이벤트로 인정하여
반대 방향 이동이나 잘못된 crossing을 제거하였다.

<p align="center">
  <img src="images/entry_crossing_qa.png" width="950">
</p>

<p align="center">
  <em>Entry Line crossing 검출 결과 QA 예시</em>
</p>

그림에서 각 sample은 차량의 전체 궤적과 함께

- Entry Line
- 진입 직전 궤적
- 진입 직후 궤적
- 실제 crossing point

를 표시한다.

QA 결과 각 Entry에서 차량이 진입선을 통과하는 시점이 정상적으로 검출되는 것을 확인하였다.

---

### 4.3 추출 결과

Location 0의 recording 02~23에서 총 **11,354개의 진입 이벤트**를 추출하였다.

| Entry | 진입 이벤트 수 |
|---|---:|
| Entry 0 | 2,685 |
| Entry 1 | 2,657 |
| Entry 2 | 3,307 |
| Entry 3 | 2,705 |
| **Total** | **11,354** |

Ego 차량 후보는 다음 motor vehicle class를 대상으로 하였다.

```text
car
van
truck
bus
trailer
```

추출된 각 진입 이벤트에는 다음 정보가 포함된다.

```text
recordingId
trackId
entryId
crossingFrame
entrySpeed
entryHeading
```

이렇게 얻은 실제 진입 시점을 기준으로 이후 단계에서
차량이 진입하기 전 여러 시점의 **Time-to-Entry(TTE)**를 계산하고,
GO / WAIT decision sample을 생성하였다.

---

## 5. GO / WAIT Label 및 Decision Sample 생성

Entry Crossing 시점이 추출된 이후에는,
각 차량이 실제로 회전교차로에 진입하기 전 여러 시점에서
의사결정 sample을 생성하였다.

핵심 기준은 **Time-to-Entry(TTE)** 이다.

TTE는 현재 시점에서 실제 Entry Crossing 시점까지 남은 시간을 의미한다.

```text
TTE = Entry Crossing Time - Current Time
```

rounD Dataset의 frame rate가 25 Hz이므로,
frame 차이를 시간으로 변환하여 TTE를 계산하였다.

---

### 5.1 Decision Sample 생성 범위

각 진입 이벤트에 대해 진입 직전 구간에서 일정 간격으로 sample을 생성하였다.

```text
TTE 범위      : 0.4 s ~ 4.0 s
Sample 간격   : 0.2 s
History 길이  : 2.0 s
History frame : 50
```

즉, 하나의 차량 진입 이벤트에 대해
진입 4.0초 전부터 0.4초 전까지 여러 개의 decision sample을 생성하였다.

이 방식으로 총 다음과 같은 sample을 구성하였다.

| Split | Sample 수 |
|---|---:|
| Train | 77,644 |
| Validation | 16,751 |
| Test | 13,296 |
| **Total** | **107,691** |

---

### 5.2 GO / WAIT Label 정의

GO / WAIT label은 실제 차량이 얼마나 가까운 시점에 진입했는지를 기준으로 정의하였다.

최종적으로 다음 기준을 사용하였다.

```text
TTE <= 1.6 s  -> GO
TTE >  1.6 s  -> WAIT
```

즉,

- **GO**: 현재 상태에서 실제 진입까지 1.6초 이하가 남은 경우
- **WAIT**: 실제 진입까지 1.6초보다 더 많이 남은 경우

로 정의하였다.

이 기준을 사용한 결과 전체 dataset의 GO / WAIT 비율이 크게 치우치지 않았으며,
Train / Validation / Test에서 유사한 분포를 유지하였다.

| Split | GO 비율 |
|---|---:|
| Train | 46.85% |
| Validation | 48.71% |
| Test | 46.55% |

---

### 5.3 Label의 의미

본 프로젝트에서 사용하는 GO / WAIT label은
안전성이 수학적으로 최적화된 진입 판단 label이 아니다.

rounD Dataset에서 실제 운전자가 보여준 행동을 기준으로,

> "현재 시점으로부터 1.6초 이내에 실제로 진입했는가"

를 나타내는 **observational behavior label**로 사용하였다.

따라서 본 모델은 운전자의 실제 진입 행동 패턴을 학습하는 것을 목표로 한다.

---

### 5.4 Pre-entry Behavior 분석

진입 전 차량의 속도 변화를 확인한 결과,
차량은 진입 시점이 가까워질수록 평균적으로 속도가 증가하는 경향을 보였다.

| 진입까지 남은 시간 | 평균 속도 |
|---|---:|
| 5 s 전 | 2.43 m/s |
| 4 s 전 | 2.57 m/s |
| 3 s 전 | 2.93 m/s |
| 2 s 전 | 3.88 m/s |
| 1 s 전 | 4.92 m/s |
| Crossing | 5.97 m/s |

또한 일부 차량은 진입 전 거의 정지한 뒤 다시 출발하는 패턴을 보였다.

```text
0.5 m/s 이하로 0.5초 이상 정지 : 18.85%
0.5 m/s 이하로 1.0초 이상 정지 : 16.25%
```

이러한 결과를 통해 단순한 현재 속도만으로는
진입 행동을 충분히 설명하기 어렵고,
과거 수 초 동안의 시계열 정보를 함께 사용하는 것이 필요하다고 판단하였다.

---

## 6. Baseline 모델 구성 및 비교

시계열 기반 회전교차로 진입 예측에서 SSM 기반 모델의 성능을 확인하기 위해
단순 kinematic baseline, LSTM, Mamba2 모델을 순차적으로 구성하여 비교하였다.

모든 학습 모델은 동일한 입력 데이터와 동일한 multi-task 출력 구조를 사용하도록 구성하여
temporal encoder 자체의 차이를 비교할 수 있도록 하였다.

---

### 6.1 Kinematic Baseline

가장 단순한 기준 모델로 현재 차량의 속도와 진행 방향이 그대로 유지된다고 가정하였다.

즉, 현재 상태만을 이용하여 향후 움직임을 선형적으로 예측하는 방식이다.

이 baseline은 과거 시계열 정보나 주변 차량과의 interaction을 사용하지 않는다.

Test 결과는 다음과 같다.

| Metric | 결과 |
|---|---:|
| Accuracy | 90.68% |
| Precision | 0.9695 |
| Recall | 0.8258 |
| F1-score | 0.8919 |
| TTE MAE | 1.6351 s |
| Entry Speed MAE | 2.1481 m/s |
| Entry Heading MAE | 15.252° |

단순한 현재 상태만으로도 일정 수준의 GO / WAIT 분류는 가능했지만,
진입 시점과 속도, 각도 예측 오차는 상대적으로 크게 나타났다.

이를 통해 회전교차로 진입 행동을 예측하기 위해서는
현재 상태뿐 아니라 과거 주행 시계열을 함께 사용하는 것이 필요함을 확인하였다.

---

### 6.2 LSTM Baseline

시계열 정보를 학습하기 위한 첫 번째 deep learning baseline으로 LSTM을 사용하였다.

구조는 다음과 같이 구성하였다.

```text
Agent history
    ↓
Shared Frame Encoder
    ↓
2-layer LSTM
    ↓
Agent Feature
    ↓
Masked Mean / Max Pooling
    ↓
Entry Geometry Feature
    ↓
Scene Representation
    ↓
Multi-task Prediction Heads
```

주요 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Hidden dimension | 64 |
| LSTM layers | 2 |
| 입력 agent 수 | 최대 9 |
| History frames | 50 |
| Model parameters | 97,670 |

LSTM 모델은 다음 네 가지 출력을 동시에 예측하도록 구성하였다.

```text
GO / WAIT
Time-to-Entry
Entry Speed
Entry Heading
```

Test 결과는 다음과 같다.

| Metric | Full LSTM |
|---|---:|
| Accuracy | 99.61% |
| F1-score | 0.9958 |
| TTE MAE | 0.1165 s |
| Entry Speed MAE | 0.3773 m/s |
| Entry Heading MAE | 2.747° |

Kinematic baseline과 비교했을 때 모든 regression metric이 크게 개선되었으며,
과거 2초의 차량 움직임을 시계열로 학습하는 것이 효과적임을 확인하였다.

---

### 6.3 Ego + Map LSTM Ablation

주변 차량 정보의 기여도를 확인하기 위해
neighbor vehicle을 제거하고 Ego trajectory와 Entry geometry만 사용하는
ablation 모델을 추가로 학습하였다.

결과는 다음과 같다.

| Metric | Ego+Map LSTM | Full LSTM |
|---|---:|---:|
| F1-score | 0.9950 | 0.9958 |
| TTE MAE | 0.1234 s | 0.1165 s |
| Entry Speed MAE | 0.3929 m/s | 0.3773 m/s |
| Entry Heading MAE | 2.705° | 2.747° |

Full LSTM이 전반적으로 더 좋은 결과를 보였지만,
Ego+Map 모델과의 차이는 크지 않았다.

따라서 단순히 주변 차량의 raw trajectory를 함께 입력하는 것만으로는
회전교차로에서 중요한 차량 간 interaction이 충분히 강조되지 않을 가능성이 있다고 판단하였다.

이 결과를 바탕으로 이후 단계에서
회전교차로 내부 conflict vehicle을 명시적으로 찾고,
TTC와 temporal gap 정보를 추가로 분석하였다.

---

### 6.4 Mamba2 Baseline

SSM 기반 temporal encoder의 성능을 확인하기 위해
LSTM과 동일한 입력 및 pooling 구조에서 Mamba2 기반 모델을 구현하였다.

Mamba2 설정은 다음과 같다.

```text
d_model = 64
d_state = 64
d_conv  = 4
expand  = 2
layers  = 2
```

전체 model parameter 수는 약 108K이다.

구조는 다음과 같다.

```text
Agent history
    ↓
Feature Projection
    ↓
Mamba2 Block
    ↓
Residual Connection
    ↓
Mamba2 Block
    ↓
Temporal Feature
    ↓
Multi-agent Pooling
    ↓
Entry Geometry
    ↓
Scene Representation
    ↓
Multi-task Heads
```

Test 결과는 다음과 같다.

| Metric | Full LSTM | Mamba2 |
|---|---:|---:|
| F1-score | 0.9958 | **0.9968** |
| TTE MAE | 0.1165 s | **0.1058 s** |
| Entry Speed MAE | 0.3773 m/s | **0.3453 m/s** |
| Entry Heading MAE | 2.747° | **2.411°** |

Mamba2는 LSTM 대비 classification 및 regression 성능에서 전반적으로 더 좋은 결과를 보였다.

특히 TTE와 Entry Heading 예측에서 개선 폭이 확인되어,
회전교차로 진입 전 시계열 패턴을 표현하는 데 SSM 기반 구조가 효과적으로 동작함을 확인하였다.


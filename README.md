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


# SME_Experiment2_Final

학번/이름 : 12233998 / 장원정

# 1. 모티베이션 & 인트로

## 중간발표까지의 실험 결과 및 고찰

중간발표에서는 팀 알고리즘 SAFL(Sensor-Agreement Fused Localization)을 공동 설계하였다. 이 알고리즘은 UWB와 WiFi 두 센서의 이종 합의도를 기반으로 앵커별 신뢰도를 산출하고(Part 2·3), 그 가중치를 Grid Search + Huber Loss + L-BFGS-B 초기 추정(Part 4) 및 IRWLS + Tukey Biweight 정제(Part 5)에 적용하는 5단계 파이프라인이었다. Test Set(N=270) 기준 MAE 1.79 m를 기록하였다.

중간발표 이후 두 가지 핵심 피드백을 받았다. 첫째, 파이프라인이 과도하게 복잡하고 불필요한 구성 요소가 포함되어 있다는 지적이었다. 특히 Part 2(이종 센서 합의도)와 Part 3(boost 기반 결합 가중치)는 최종 데이터셋에 UWB와 WiFi 센서 구분이 존재하지 않아 물리적 근거가 성립하지 않는다는 점이 핵심 문제였다. 둘째, 각 구성 요소가 실제로 성능에 기여하는지 검증이 필요하다는 지적이었다.

이에 따라 Part 2와 Part 3을 제거하고 나머지 구조를 유지하여 성능을 재측정하였다. 그 결과 MAE 9.55 m를 기록하였으며, 이는 중간발표 당시보다 크게 저하된 수치였다. 저하 원인을 분석하기 위해 데이터를 직접 분석하였다.

학습 데이터 700명에 대해 18개 앵커 각각의 RTT 측정값과 GT 위치 기반 실제 거리의 차이(NLOS bias)를 계산한 결과, 앵커별 중앙값 bias가 6.37 m에서 13.14 m 수준으로 앵커마다 최대 2배 가까이 차이가 났다. 전체 중앙값 bias는 10.07 m였다. 앵커별 고정 median bias를 차감하는 방식은 위치에 따라 달라지는 NLOS 경로 특성을 반영하지 못해, 보정 후에도 잔존 오차의 중앙값이 9.94 m에 달하였다.

이 분석에서 bias를 명시적으로 추정하는 접근의 근본적 한계를 확인하였다. 위치마다 NLOS 경로가 달라지므로 학습 데이터 전체의 통계량으로 구한 고정 bias는 테스트 시점에서 과보정 혹은 과소보정 문제를 피할 수 없다.

## 새로운 관점: NLOS 물리 성질의 직접 활용

데이터 분석 과정에서 NLOS 환경의 중요한 물리적 사실을 주목하게 되었다. NLOS에서는 신호가 반사 경로를 통해 도달하므로 RTT 측정값이 실제 거리보다 항상 크거나 같다. 즉 d_hat[i] >= d_true[i] = ||x_true - b_i||가 성립한다. 이는 실제 사용자 위치가 18개 측정 원 D_i = {q : ||q - b_i|| <= d_hat[i]}의 교집합 내부에 반드시 존재한다는 기하학적 사실로 이어진다.

이 성질을 이용하면 bias를 추정하지 않고도 위치를 찾을 수 있다. 모든 측정 원의 교집합에서 가장 깊은 내부점(Chebyshev Center)을 구하는 Minimax 최적화가 이 아이디어의 핵심이다. Chebyshev Center는 학습 데이터를 전혀 사용하지 않으므로 완전 비지도이고, NLOS 환경이 바뀌어도 기하학적 성질이 유지되는 한 일반화 성능이 안정적이다.

## 알고리즘 아이디어 도출 흐름 및 high-level 소개

Minimax 단독으로 MAE 6.73 m, 2 m 이내 35.3%를 확인하였다. bias 보정 기반 방법보다 우수한 결과였다. 여기에 LTS(Least Trimmed Squares) 정밀화를 추가하자 Median이 5.27 m에서 1.26 m로 크게 개선되었고, 2 m 이내 비율이 58.7%로 상승하였다. LTS가 실패하는 케이스를 Slack 기반 Fallback으로 방지한 것이 최종 구조이다.

최종 알고리즘은 세 단계로 구성된다. Step 1에서는 NLOS 양의 편향 성질을 이용한 Minimax로 초기 위치를 추정하고, Step 2에서는 LTS로 잔차 하위 k개 앵커만 fit하여 정밀화하며, Step 3에서는 Slack 기반 Fallback으로 이상 결과를 방지한다. GT 위치를 일체 사용하지 않는 완전 비지도 방법이다.

---

# 2. 알고리즘 설명

입력은 UE 한 명에 대한 RTT 기반 측정 거리 벡터 d = (d_1, ..., d_18)와 18개 기지국 좌표 B = {b_i ∈ R^2 | i = 1, ..., 18}이다. 출력은 추정 위치 x_hat ∈ R^2이다.

## Step 0. 탐색 공간 설정 (오프라인, 1회)

기지국 좌표의 x, y 방향 범위 W_x, W_y를 계산하고 각 방향으로 30%를 마진으로 추가하여 탐색 경계를 설정한다.

B_x = [x_min - 0.3 * W_x, x_max + 0.3 * W_x]

B_y = [y_min - 0.3 * W_y, y_max + 0.3 * W_y]

이 경계 내부에 0.5 m 간격의 2차원 격자 G를 구성한다. 격자는 기지국 좌표만으로 결정되며 학습 데이터가 필요 없다. main() 진입 시 1회 계산하여 모든 UE 추론에 재사용한다.

## Step 1. Minimax (Chebyshev Center)

NLOS 환경에서는 d_hat[i] >= d_true[i]가 성립하므로 실제 위치는 18개 측정 원 D_i = {q : ||q - b_i|| <= d_i}의 교집합 내부에 존재한다.

후보 위치 q에서 기지국 i에 대한 slack을 다음과 같이 정의한다.

s_i(q) = ||q - b_i|| - d_i

s_i < 0이면 q가 원 i의 내부에 있음을 의미하고, s_i > 0이면 원 밖으로 벗어난 위반을 나타낸다. Chebyshev Center는 최악의 위반을 최소화하는 점이다.

f(q) = min_q max_i (||q - b_i|| - d_i)

max 연산은 미분 불가능하므로 다음의 log-sum-exp로 smooth하게 근사한다. alpha = 10으로 설정하면 max에 충분히 가까운 근사를 제공하면서 L-BFGS-B 적용이 가능해진다.

f_smooth(q) = (1 / alpha) * log( sum_i exp(alpha * s_i(q)) )

격자 G의 각 점에서 f_smooth를 평가하여 값이 가장 작은 격자점을 초기점 q_0으로 선택한다. 이후 L-BFGS-B로 탐색 경계 B_x × B_y 내에서 수치 최소화하여 p_minimax를 확정한다.

## Step 2. Least Trimmed Squares (LTS) 정밀화

p_minimax는 모든 원 내부의 안전한 점이지만 실제 위치와 수 m 거리가 있을 수 있다. LTS는 18개 앵커 잔차 중 제곱값이 가장 작은 k = 7개만의 합을 최소화하여 위치를 정밀화한다.

각 앵커의 잔차를 r_i(q) = ||q - b_i|| - d_i로 정의한다. 잔차 제곱값을 오름차순으로 정렬한 r_(1)^2 <= r_(2)^2 <= ... <= r_(18)^2에 대해 LTS 목적함수는 다음과 같다.

L_LTS(q) = sum_{j=1}^{7} r_(j)^2(q)

k = 7은 18개 중 잔차가 작은 7개 앵커에만 fit하고, NLOS가 심한 나머지 11개 앵커를 자동으로 무시하는 효과를 가진다. p_minimax를 초기점으로 L-BFGS-B로 최소화하여 p_lts를 확정한다.

## Step 3. Slack 기반 Fallback

LTS 정밀화 과정에서 강한 NLOS 앵커 쪽으로 끌려가 측정 원 밖으로 이탈할 수 있다. p_lts에서 모든 앵커에 대한 slack의 최댓값을 계산한다.

max_slack = max_i (||p_lts - b_i|| - d_i)

max_slack > 5 m이면 p_lts가 NLOS 물리 성질(d_hat >= d_true)을 위반한 것으로 판단하여 p_minimax를 최종 추정값으로 반환한다. max_slack <= 5 m이면 p_lts를 최종 추정값으로 반환한다.

## 비지도 특성 요약

본 알고리즘은 어떤 단계에서도 GT 위치(p)를 사용하지 않는다. 탐색 공간은 기지국 좌표만으로 결정되고, Minimax와 LTS는 측정값 d_hat과 기지국 좌표 BS_positions만을 입력으로 사용한다. Fallback 기준인 5 m 역시 측정값과 기하 거리의 일관성만으로 판단한다.

---

# 3. Agent AI 활용 방안

본 프로젝트에서는 Claude(Anthropic)를 주요 Agent AI로 활용하였다.

## 본인의 역할

중간발표 피드백을 분석하여 방향을 재정의하는 판단, 어떤 실험을 수행할지 결정하는 설계, 각 실험 결과의 수치를 해석하여 다음 방향을 결정하는 과정, NLOS 양의 편향 성질에서 Minimax 아이디어를 도출하는 통찰, LTS의 k값과 Slack Fallback의 threshold를 결정하는 판단은 모두 본인이 직접 수행하였다.

## Claude의 역할

실험 코드 구현 및 성능 측정 실행, 삼각부등식·LMedS·LTS·Rank-based 등 다양한 방법론의 비교 실험, 수식 변형(v1~v4) 성능 비교, 최종 main.py 코드 작성 및 README 규격 검토는 Claude가 담당하였다.

## 역할 구분 표

| 구분 | 본인 | Claude |
|---|---|---|
| 문제 재정의 | NLOS 양의 편향 성질 발견 및 Minimax 아이디어 도출 | 여러 방법론 설명 및 구현 제안 |
| 실험 설계 | 어떤 방법을 시도할지 결정 | 코드 작성 및 실행 |
| 결과 해석 | 수치 기반 방향 판단 | 수치 정리 및 비교표 작성 |
| 코드 작성 | 알고리즘 설계 방향 지시 | main.py 구현 |
| 보고서 | 실험 흐름 및 설계 의도 제공 | 문장 구성 보조 |

## 구체적 활용 사례

삼각부등식 기반 outlier 제거를 시도했을 때, Claude가 "d_corr 표준편차 32.9 m, 앵커 간 최대 거리 107.7 m, LOS 앵커를 38.4% 잘못 제거"라는 진단 수치를 제시하였다. 이 수치를 보고 본인이 삼각부등식 접근의 근본 한계를 판단하고 Minimax 방향으로 전환을 결정하였다.

Minimax 수식 변형 실험에서는 현재 수식(v1), 절댓값 수식(v2), max(0, 원 외부 위반)(v3), max(0, 측정-거리)(v4) 네 가지를 Claude가 구현하고 성능을 측정하였다. 팀원 피드백에서 v4가 더 자연스럽다는 의견이 있었으나, 실험 결과 v4가 MAE 54.59 m로 가장 나빠 v1을 유지하기로 본인이 판단하였다.

---

# 4. 결과 도출 & 디스커션

## 자체 평가 방식

제공된 학습 데이터 700명 전체를 평가에 사용하였다. 본 알고리즘은 완전 비지도이므로 GT 위치를 일체 사용하지 않아 학습/테스트 분리가 필요 없다. 700명 전체에서 평가한 결과가 hidden test set에서의 성능을 공정하게 근사한다.

평가 지표는 MAE(평균 절대 오차), Median Error(50번째 백분위수), RMSE(제곱평균제곱근 오차), P90(90번째 백분위수), 2 m 이내 비율이다.

## Baseline 비교 설계

비교 대상을 모두 비지도 또는 통계량만 사용하는 방법으로 구성하였다. 단계별 ablation을 통해 각 구성 요소의 기여도를 분리하여 분석함으로써 비교의 공정성을 확보하였다.

| 방법 | 설명 | GT 사용 여부 |
|---|---|---|
| Simple LS | raw d_hat으로 최소제곱 삼변측량 | 없음 |
| Bias보정 + LS | 앵커별 median bias 차감 후 LS | 통계량만 |
| SAFL (중간발표) | Bias보정 + Grid + Huber + IRWLS | 통계량만 |
| Minimax only | Step 1만 수행 (ablation) | 없음 |
| 제안 알고리즘 (Full) | Step 1 + Step 2 + Step 3 | 없음 |

## 실험 결과

| 방법 | MAE (m) | Median (m) | RMSE (m) | P90 (m) | 2m 이내 (%) |
|---|---:|---:|---:|---:|---:|
| Simple LS (raw) | 39.352 | 27.054 | 61.542 | 72.629 | 0.3 |
| Bias보정 + LS | 32.479 | 21.106 | 53.670 | 58.971 | 0.4 |
| SAFL (중간발표) | 9.553 | 8.195 | 11.790 | 17.664 | 3.6 |
| Minimax only (Step 1) | 6.727 | 5.270 | 9.210 | 15.324 | 35.3 |
| 제안 알고리즘 (Full) | 4.800 | 1.261 | 7.990 | 14.714 | 58.7 |

## 결과 해석

Simple LS는 NLOS bias를 전혀 처리하지 않아 MAE 39.35 m로 가장 나쁘다. Bias보정 + LS는 앵커별 고정 bias를 제거하지만 위치 의존적 NLOS 변동을 반영하지 못해 개선 폭이 제한적이다.

SAFL은 Huber Loss와 IRWLS로 outlier를 억제하여 MAE 9.55 m를 달성하지만, bias 보정 후에도 잔존 오차가 크다는 한계가 있다. 이는 앵커별 고정 bias 가정의 한계에서 비롯된다.

Minimax only는 bias 추정 없이 NLOS 물리 성질만으로 MAE 6.73 m, 2 m 이내 35.3%를 달성한다. bias 보정 기반인 SAFL보다 오히려 우수한 이유는 Chebyshev Center가 LOS에 가까운 앵커들의 교집합 방향으로 자연스럽게 수렴하기 때문이다.

LTS 정밀화(Step 2)를 추가하면 Median이 5.27 m에서 1.26 m로 크게 개선되고 2 m 이내 비율이 58.7%로 상승한다. LTS는 잔차 하위 7개 앵커에만 fit하므로 NLOS가 심한 나머지 앵커를 자동으로 무시한다. Slack Fallback(Step 3)은 LTS가 실패하는 샘플에서 폭발을 방지하여 MAE 안정성을 확보한다.

## 비교의 공정성

본 알고리즘은 GT 위치를 전혀 사용하지 않으므로 Bias보정 계열과의 비교에서도 불리한 조건이다. 그럼에도 불구하고 SAFL보다 MAE와 Median 모두 우수하였다. GT를 전혀 사용하지 않는 Simple LS와 비교하면 MAE 39.35 m에서 4.80 m로 개선되었으며, 이것이 알고리즘의 실질적 기여를 보여주는 가장 fair한 비교이다.

## 알고리즘 장점

완전 비지도 구조로 학습 데이터 과적합 위험이 없다. hidden test set의 NLOS 환경이 학습 데이터와 달라도 성능 저하가 구조적으로 제한된다. NLOS 양의 편향이라는 물리적 사실만을 가정하므로 환경 의존적 hyperparameter가 적다. 1000명 처리 시간이 약 15초로 10분 제한 대비 40배 여유가 있다.

## 알고리즘 단점

LTS의 k = 7은 18개 중 절반 이하가 NLOS outlier라는 가정에 기반한다. 극단적 NLOS 환경에서 outlier 앵커가 9개 이상이면 LTS가 실패할 수 있으며, Fallback이 보완하지만 완전한 해결은 아니다. P90이 14.71 m로 일부 샘플에서 큰 오차가 잔존한다.

## Future Work

k를 고정하는 대신 Minimax 결과에서 잔차 분포를 분석하여 샘플별로 동적으로 결정하는 방법을 고려할 수 있다. Slack Fallback threshold를 고정값 대신 해당 샘플의 d_hat 중앙값에 비례하여 설정하면 다양한 환경에서 더 강건한 판단이 가능하다.

---

# 5. Reference

[1] P. C. Chen, "A non-line-of-sight error mitigation algorithm in location estimation," Proc. IEEE WCNC, 1999.

이 논문은 NLOS 환경에서 RTT 측정값이 실제 거리보다 크게 측정되는 현상을 분석하고, 이를 통계적으로 식별하여 완화하는 알고리즘을 제안한다. 구체적으로는 측정 잔차의 분포를 분석하여 NLOS 여부를 hypothesis test로 판별한 뒤, 보정 계수를 적용하는 방식이다.

본 알고리즘이 참고한 부분은 NLOS 측정값이 실제 거리보다 크다는 물리적 관찰이다. 이 관찰로부터 d_hat[i] >= d_true[i]라는 부등식 성질을 착안하였고, 이것이 Minimax 목적함수 설계의 출발점이 되었다. 차이점은 이 논문이 NLOS를 이진 분류하거나 보정 계수를 추정하는 방식인 반면, 본 알고리즘은 NLOS 여부를 판별하지 않고 d_hat >= d_true라는 성질을 목적함수에 직접 반영하여 Chebyshev Center를 탐색한다는 점이다. 따라서 GT 위치 없이도 동작하는 비지도 구조가 가능하다.

[2] S. Maranò, W. M. Gifford, H. Wymeersch, and M. Z. Win, "NLOS Identification and Mitigation for Localization Based on UWB Experimental Data," IEEE Journal on Selected Areas in Communications, vol. 28, no. 7, pp. 1026-1035, 2010.

이 논문은 UWB 실험 데이터를 기반으로 NLOS 측정값이 양의 편향(positive bias)을 가지며 분산이 크다는 통계적 특성을 보이고, 이를 식별하여 위치 추정 정확도를 높이는 방법을 제안한다. NLOS 측정값을 soft constraint, 즉 단방향 부등식 제약으로 처리하는 아이디어도 제시한다.

본 알고리즘이 참고한 부분은 NLOS 측정값을 상한(upper bound) 제약으로 해석하는 관점이다. 이 논문이 d_hat[i]를 ||x - b_i||의 상한으로 보는 soft constraint 개념을 제안한 것에서, 실제 위치가 모든 측정 원의 내부에 존재한다는 기하학적 해석을 이끌어내었다. 차이점은 이 논문이 NLOS를 먼저 식별(classification)한 뒤 outlier를 제거하거나 재가중하는 2단계 방식인 반면, 본 알고리즘은 NLOS 식별 과정 없이 모든 측정값을 상한 제약으로 간주하여 Minimax 최적화를 수행한다. 또한 정밀화 단계에서 이 논문의 부등식 제약 방식 대신 LTS를 사용하여 잔차 작은 앵커만 fit하는 방식으로 outlier를 처리한다.

[3] P. J. Huber, "Robust Estimation of a Location Parameter," Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73-101, 1964.

이 논문은 M-estimator 개념을 도입하여 outlier에 강건한 위치 추정 방법론의 이론적 토대를 제시한다. Huber Loss는 잔차가 작을 때 제곱 손실, 클 때 선형 손실을 적용하여 outlier의 영향을 제한한다.

본 알고리즘이 참고한 부분은 outlier를 완전히 제거하지 않고 잔차 크기에 따라 차등 취급하는 robust estimation의 철학이다. 중간발표 SAFL 알고리즘의 Part 4에서 Huber Loss를 적용하여 outlier 앵커의 영향을 억제하였다. 최종 알고리즘에서는 Huber Loss 대신 LTS를 채택하였는데, LTS는 잔차 하위 k개만 fit하여 나머지를 완전히 무시한다는 점에서 Huber의 연속 가중치 방식보다 더 강한 outlier 배제가 가능하다. 이 선택은 bias 보정 없이 raw d_hat을 사용하는 본 알고리즘에서 잔차 분포가 Huber 전환점(5 m)을 훨씬 넘는 경우가 많아 Huber보다 LTS가 더 적합하다는 실험적 확인에 근거한다.

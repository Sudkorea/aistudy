마르코프 체인(Markov Chain)에 대해 수학적으로 상세히 설명하겠음.

마르코프 성질(Markov property)을 만족하는 확률 과정(stochastic process)임. 즉, 미래 상태의 조건부 확률 분포가 과거 상태와 독립적이며 오직 현재 상태에만 의존함.

## 수학적 정의

확률 공간 (Ω, F, P)에서 상태 공간 S를 가지는 이산 시간 확률 과정 {X_n : n ≥ 0}에 대해, 모든 n ≥ 0와 모든 상태 $i_0, ..., i_n, i_{n+1} ∈ S$에 대해 다음이 성립하면 마르코프 체인이라 함:

$$ P(X_{n+1} = i_{n+1} | X_n = i_n, X_{n-1} = i_{n-1}, ..., X_0 = i_0) = P(X_{n+1} = i_{n+1} | X_n = i_n) $$

## 전이 확률 (Transition Probability)

상태 i에서 상태 j로의 한 단계 전이 확률:

$$ p_{ij} = P(X_{n+1} = j | X_n = i) $$

전이 확률 행렬 P = (p_{ij})는 각 행의 합이 1인 확률 행렬임.

## n-단계 전이 확률

n 단계 후 상태 i에서 상태 j로 전이할 확률:

$$ p_{ij}^{(n)} = P(X_{n+m} = j | X_m = i) $$

Chapman-Kolmogorov 방정식:

$$ p_{ij}^{(m+n)} = \sum_{k \in S} p_{ik}^{(m)} p_{kj}^{(n)} $$

이는 행렬 표기로 P^(m+n) = P^m P^n로 표현됨.

## 정상 분포 (Stationary Distribution)

π가 마르코프 체인의 정상 분포라면 다음을 만족함:

$$ \pi = \pi P $$

즉, $π_j = Σ_i π_i p_{ij}$ 모든 j에 대해 성립.

## 에르고딕성 (Ergodicity)

마르코프 체인이 에르고딕하다면, 초기 분포와 무관하게 장기적으로 고유한 정상 분포에 수렴함. 에르고딕성의 조건:

1. 비주기성 (Aperiodic): 모든 상태의 주기가 1
2. 양재귀성 (Positive Recurrent): 모든 상태로 돌아올 평균 시간이 유한
3. 연결성 (Irreducible): 모든 상태 쌍 사이에 양의 확률을 가진 경로가 존재

## 상세 균형 조건 (Detailed Balance)

만약 π와 P가 다음을 만족하면, π는 P의 정상 분포임:

$$ \pi_i p_{ij} = \pi_j p_{ji} \quad \forall i,j \in S $$

이는 MCMC 알고리즘 설계에서 중요한 역할을 함.

## 극한 정리 (Limit Theorems)

에르고딕 마르코프 체인에 대해:

1. 수렴 정리: $\displaystyle\lim_{n→∞} p_{ij}^{(n)} = π_j$
2. 에르고딕 정리: $\displaystyle\lim_{n→∞} {\frac{1}{n}} \sum_{k=1}^n f(X_k) = \sum_i π_i f(i)$ (확률 1)

# 마르코프 체인의 응용

1. MCMC (Markov Chain Monte Carlo):
   - Metropolis-Hastings 알고리즘: 제안 분포 q(y|x)를 사용하여 마르코프 체인을 구성
     $$ α(x,y) = min(1, \frac{π(y)q(x|y)}{π(x)q(y|x)}) $$
   - Gibbs Sampling: 조건부 분포를 사용하여 각 변수를 순차적으로 갱신

2. 은닉 마르코프 모델 (HMM):
   - 관측 불가능한 상태 시퀀스가 마르코프 체인을 따름
   - 전방-후방 알고리즘, Viterbi 알고리즘 등에 활용

3. 페이지랭크 (PageRank):
   - 웹 페이지의 중요도를 마르코프 체인의 정상 분포로 모델링

4. 큐잉 이론:
   - 시스템의 상태 변화를 마르코프 체인으로 모델링

5. 금융 모델링:
   - 자산 가격 변동이나 신용 등급 변화를 마르코프 체인으로 표현

6. 생물정보학:
   - DNA 서열 분석, 단백질 구조 예측 등에 활용

7. 강화학습:
   - 환경과 에이전트의 상호작용을 마르코프 결정 과정(MDP)으로 모델링


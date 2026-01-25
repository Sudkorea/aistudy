# 위상적 데이터 분석 (Topological Data Analysis, TDA)

## 1. 서론

위상적 데이터 분석(Topological Data Analysis, TDA)은 **대수적 위상수학(algebraic topology)** 과 **계산기하학(computational geometry)** 을 응용하여 데이터의 전역적 구조를 파악하는 기법이다.  
특히 데이터가 단순히 수치적 요약 통계로는 설명하기 어려운 **형태(shape)** 와 **연결성(connectivity)** 을 가진다고 볼 때, TDA는 이를 **위상적 불변량(topological invariant)** 을 통해 정량화할 수 있다.  
또한 TDA는 노이즈에 강건(stability theorem 존재)하며, 고차원 데이터, 비정형 데이터, 혼합형 데이터에 적용이 가능하여 최근 다양한 분야(의료, 금융, 네트워크 분석 등)에서 활용되고 있다.

---

## 2. 기본 개념

### 2.1 심플렉스와 단순복합체

- **$k$-심플렉스(simplex)**: $k+1$개의 아핀 독립$^*$ 점의 convex hull.
    
    - 0-simplex: 점, 1-simplex: 선분, 2-simplex: 삼각형, 3-simplex: 사면체.
        ![[Pasted image 20250918163954.png]]
- **심플리셜 컴플렉스(simplicial complex)**: 심플렉스들의 모임으로, 어떤 심플렉스가 포함되면 그 부분심플렉스도 반드시 포함되어야 한다.
  ![[Pasted image 20250918164127.png]]
  (torus는 simplicial complex로 나타낼 수 있음.)
* 아핀 독립 : 모든 점이 남은 점들로 생성된 부분 아핀 공간에 속하지 않는 아핀 공간 속 점들의 집합

### 2.2 필트레이션 (Filtration)

스케일 파라미터 $\varepsilon$을 변화시키며 생성되는 **증가하는 복합체의 사슬**:

$$\emptyset = K_0 \subseteq K_1 \subseteq \cdots \subseteq K_m$$
![[Pasted image 20250918164721.png]]
데이터의 연결 구조가 $\varepsilon$에 따라 어떻게 진화하는지를 나타낸다.

- 작은 $\varepsilon$: 점들이 고립된 상태
    
- $\varepsilon$ 증가: edge, face 추가 → 더 복잡한 구조 형성
    

---

## 3. 호몰로지 (Homology)

### 3.1 체인 복합체 (Chain Complex)

심플리셜 컴플렉스 $K$에 대해:

- $C_k(K)$: $k$-심플렉스들의 자유 아벨 군$^*$
  (\*아벨 군 : 교환 법칙을 만족하는 Group)
    
- 경계 사상(boundary map):
  심플리셜 컴플렉스 $K$의 경계를 뱉는 연산
    
    $$\partial_k: C_k(K) \to C_{k-1}(K), \quad \partial_k \circ \partial_{k+1} = 0$$

### 3.2 사이클과 경계

- 사이클 그룹: $Z_k = \ker \partial_k$
    ![[Pasted image 20250918170427.png]]
- 경계 그룹: $B_k = \operatorname{im} \partial_{k+1}$
    ![[Pasted image 20250918170608.png]]
### 3.3 몫공간(Quotient space)으로서의 호몰로지

$H_k(K) = Z_k / B_k$

즉, 모든 사이클 중 경계로 해소될 수 있는 것들은 동일하게 취급하고, 남는 것이 비자명한 “구멍”이다.

#### 여기서 Quotient space란?

우선, **벡터공간(vector space)** $V$란, 원소의 집합이 벡터 덧셈과 스칼라 곱셈에 대해 닫혀 있으며, 교환법칙·결합법칙·항등원과 역원 존재·분배법칙 등을 만족하는 구조를 말한다. 예를 들어, $\mathbb{R}^n$은 표준적인 벡터공간이다.
![[Pasted image 20250919122410.png]]
(선형대수학에서의 벡터공간의 정의)

벡터공간 $V$ 안에 부분공간(subspace) $W \subseteq V$가 있다고 하자. 이때 우리는 $V$의 원소들을 **“$W$ 만큼의 차이를 무시한다”** 는 관점으로 묶을 수 있다.  
구체적으로, $v_1, v_2 \in V$에 대해

$$v_1 \sim v_2 \quad \Longleftrightarrow \quad v_1 - v_2 \in W$$
라는 동치관계를 정의한다.  
즉, 두 벡터의 차이가 $W$ 안에 속한다면 같은 것으로 간주한다.

이 동치류들의 집합 $V/W$를 **몫공간(quotient space)** 라고 한다. 각 원소는 “$v + W$” 꼴의 집합(즉, $v$와 $W$에 속하는 모든 벡터를 더한 집합)으로 표현된다. 이를 **coset**이라고 부른다.

- 그나마 직관적인 예시 : 정수 중 3으로 나누어 떨어지는 원소들
  
  정수 중 3으로 나누어떨어지는 원소의 집합 $3 \mathbb {Z} = \{ \dots , -3, 0, 3, 6, \dots \}$ 는 정수 집합 $\mathbb {Z}$ 의 부분집합이며, 위의 사진에서 Vector Space 조건을 모두 만족한다.(직접 종이 위에 조건을 만족하는지 적어보면, 훨씬 이해에 도움이 된다.) $\rightarrow$ 집합 $3 \mathbb {Z}$는 subspace이다!
  
  이제, 위의 동치류 $\sim$를 생각해 보면(동치가 말이 어려운데, 대강 같다고 보면 된다.)
  $$3 \sim 0 \quad \Longleftrightarrow \quad 3 - 0 = 3 \in 3 \mathbb {Z}$$
  $$6 \sim 3 \quad \Longleftrightarrow \quad 6 - 3 = 3 \in 3 \mathbb {Z}$$
  우선, 3의 배수에 대해서는 잘 만족한다. 그렇다면, 1은?
  $$3 \nsim 1 \quad \Longleftrightarrow \quad 3 - 1 = 2 \notin 3 \mathbb {Z}$$
  그렇다면, 1의 동치류를 찾으면 어떤 것들이 있을까?
  $$1 \sim 4 \quad \Longleftrightarrow \quad 1 - 4 = -3 \in 3 \mathbb {Z}$$
  위와 같이, 3으로 나누면 1이 남는 모든 정수는 1과 동치일 것이다.
  2의 동치류 또한 자명하게, 나누었을 때 나머지가 2인 모든 정수와 동치이다. 
  
  따라서, 모든 정수는 3으로 나누었을 때 (나누어 떨어지는 / 나머지가 1인 / 나머지가 2인) 세 그룹으로 나뉘어지고, 이것을 quotient space로 나타내면
  $$\mathbb {Z}/3 \mathbb {Z} = \{3 \mathbb {Z},1 + 3 \mathbb {Z}, 2+3 \mathbb {Z}\}$$
  으로 표현되고, 이를 눈에 보이게 나타내면 다음과 같다.
  ![[Pasted image 20250919123257.png]]
  (같은 색은 같은 coset 안의 원소이다.)
  
- **그래서 이걸 왜 사용하나?**
  한줄정리 : 원소가 아닌 집합에도 vector space라는 강력한 성질을 부여하고 갖고놀기 위해
  
  모든 정수는 3의 배수라는 부분집합으로 자르면 3가지 특성을 가진 집합으로 분해되며, 이 세 집합은 또 vector space의 조건을 만족하기 때문에, 원소가 아닌 집합의 구조에도 선형대수에 쓰이는 모든 정리들을 갖다쓸 수 있다는 강력한 효과가 있어 이 집합을 정의한다.
  
  
몫공간 $V/W$는 새로운 벡터공간 구조를 가지며, 그 차원은

$$\dim(V/W) = \dim(V) - \dim(W)$$

으로 주어진다.

호몰로지에서

$$H_k(K) = Z_k / B_k$$

라는 정의는, $Z_k$ (사이클 그룹)라는 벡터공간에서 $B_k$ (경계 그룹)이라는 부분공간을 “무시한” 몫공간을 의미한다. 즉, 모든 사이클을 고려하되, 단순히 경계로부터 온 것들은 동치로 취급하고, 남는 요소만을 독립된 구멍으로 본다는 것이다.

### 3.4 Betti 수

- $\beta_0$: 연결 성분 개수
    
- $\beta_1$: 독립적인 루프 개수
    
- $\beta_2$: 2차원 void 개수
![[Pasted image 20250919132713.png]]

---
# 4. 지속 호몰로지 (Persistent Homology)

### 4.1 개념

- 데이터를 일정한 스케일(ε)로 확장해가며 **연결 성분(β₀), 루프(β₁), void(β₂)** 같은 구조가 어떻게 **생기고(birth)**, **사라지는(death)** 지를 추적한다.
    
- 짧게 나타났다 사라지는 구조는 **노이즈**, 오래 지속되는 구조는 **데이터의 본질적 특성**으로 본다.
    

---

### 4.2 과정

![[Pasted image 20250919133646.png]]
1. **점구름(point cloud)**: 데이터는 n차원 공간의 점들로 놓임.
    
2. **Filtration 생성**: ε을 증가시키며 점들을 연결 → simplicial complex 생성.
    
3. **Homology 계산**: 각 단계에서 β₀, β₁, β₂ 변화를 기록.
    
4. **Birth–Death 추적**: 새로운 구조가 생기는 순간(birth), 소멸하는 순간(death)을 기록.
    

---

### 4.3 시각화
![[Pasted image 20250919134045.png]]
- **Barcode**: 각 구조의 수명(lifetime)을 막대기로 표시.
    
    - 긴 막대: 의미 있는 패턴
        
    - 짧은 막대: 잡음 가능성
        
- **Persistence diagram**: birth–death 쌍을 평면 위 점으로 표시.
    
    - 대각선과 멀리 떨어진 점일수록 구조적으로 의미 있음.
        

---

### 4.4 해석

- 데이터의 **형태(shape)** 를 수치적으로 요약할 수 있음.
    
- 예:
    
    - β₀: 클러스터 수
        
    - β₁: 원형/루프 패턴 존재 여부
        
    - β₂: 3차원 void 여부
        

---

# 5. 응용 사례 (대표 연구 분석)

## 5.1 생물학 — 단백질 구조·접힘(folding)

**Xia & Wei (2014)**: _Persistent homology analysis of protein structure, flexibility, and folding_

- **데이터/문제**: 단백질의 원자 좌표(전원자·coarse-grained)로부터 구조적 특징·유연성(B-factor)·접힘 과정의 위상 변화를 정량화.
    
- **방법**: 원자들을 point cloud로 두고 거리/상호작용을 반영해 filtration(상관행렬 기반 포함) → **persistent homology**로 분자 위상 지문(MTF) 추출 → 누적 막대 길이 등으로 **유연성/안정성** 지표화.
    ![[Pasted image 20250922105311.png]]
- **결과/의의**: PH 기반 지표가 단백질 **compactness/rigidity**와 정량적으로 상관, 접힘 단계의 위상적 진화를 포착. 기존 물리 모델(ENM 등)의 **특성 거리 선택**에도 통찰 제공. 분자 수준에서 “형태–기능” 연결을 **위상적 요약**으로 제시. ([arXiv](https://arxiv.org/abs/1412.2779?utm_source=chatgpt.com "Persistent homology analysis of protein structure, flexibility and folding"))
    

---

## 5.2 의학 — EEG/fMRI 신호(발작 탐지·뇌기능 네트워크)

**Wang et al. (2023)**: _Automatic epileptic seizure detection based on persistent homology_

- **데이터/문제**: 임상 **EEG**에서 발작(ictal) vs 비발작( inter-ictal) 구간 자동 분류.
    
- **방법**: EEG를 윈도우로 분할 → 각 창에 대해 **Vietoris–Rips** 복합체 filtration → **diagram/바코드** → 벡터화(예: persistence 통계) → 분류기.
    ![[Pasted image 20250922105405.png]]
- **결과/의의**: PH 특성이 발작 시의 **비정상 위상 신호**(연결성/루프 구조 변조)를 포착해 높은 분류 성능을 달성. PH 기반 파이프라인을 임상 신호에 **엔드-투-엔드**로 적용한 최근 사례. 보조로 fMRI 네트워크의 PH 안정성·통계 추론 연구들이 병행되어 신뢰성 근거 강화. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10773586/?utm_source=chatgpt.com "Automatic epileptic seizure detection based on persistent ..."))
    

> (보충) **Connectome 네트워크**에서의 고차 구조: **Sizemore et al.** 은 인간 구조연결망에서 **클리크와 cavity**를 검출, 고차 공동구조가 정보처리에 기여함을 보였음(네트워크-PH 관점의 의학/신경과학 교차 레퍼런스). ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5769855/?utm_source=chatgpt.com "Cliques and cavities in the human connectome - PMC"))

---

## 5.3 금융 — 시계열 레짐 변화/위기 조기징후

**Gidea & Katz (2017/2018)**: _Topological Data Analysis of Financial Time Series: Landscapes of Crashes_

- **데이터/문제**: 미국 주요 지수 일간수익률(다변량 시계열)에서 **버블/붕괴 전조** 탐지.
    
- **방법**: 시계열 **sliding window** → 임베딩된 point cloud에 Rips-PH → **persistence landscape**를 시간축으로 추적 → $\ell^p$-norm 시계열을 지표화.
    ![[Pasted image 20250922105539.png]]
    ![[Pasted image 20250922105554.png]]
- **결과/의의**: 닷컴(2000), 리먼(2008) 전 **수백 거래일**에 걸쳐 landscape $\ell^p$-norm의 **유의한 상승 추세** 포착 → 전통 통계로 잡기 어려운 **구조 전이 신호**를 위상 특성으로 조기 감지 가능성을 제시. 실증·모형 데이터 모두에서 재현. ([arXiv](https://arxiv.org/abs/1703.04385?utm_source=chatgpt.com "Topological Data Analysis of Financial Time Series: Landscapes of Crashes"))
    

---

## 5.4 추천 시스템 — 사용자–아이템 구조의 위상 피처 활용

**Falih et al. (2018)**: _Topological multi-view clustering for collaborative filtering (MovieLens)_

- **데이터/문제**: **MovieLens** 추천 데이터에서 다중 뷰(평점, 메타정보 등)를 결합해 군집/추천 품질 향상.
    
- **방법**: 각 뷰에서 그래프/거리 기반 구조를 만들고 **topological multi-view clustering**(필터/클러스터 병합 및 위상적 일관성 고려)으로 사용자/아이템의 잠재 구조를 반영.
    
- **결과/의의**: 다중 뷰의 **위상적 합성**이 군집 품질과 추천 성능 개선에 기여함을 실증(협업필터링 보강). 추천 영역에서 **TDA-기반 표현**이 실제 벤치마크에서 동작함을 보인 초기 사례. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050918322336/pdf?md5=7d4874eb9ba70880e2bdbdd3baf6b159&pid=1-s2.0-S1877050918322336-main.pdf&utm_source=chatgpt.com "Topological multi-view clustering for collaborative filtering"))
    ![[Pasted image 20250922105813.png]]

> (참고) 최근엔 **Mapper**로 고차 임베딩을 그래프로 요약해 세그먼트·편향을 설명하는 시각 분석 흐름도 늘어나는 중. 실무 적용 시 파라미터 의존성에 주의(가이드 논문 다수). ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614807/?utm_source=chatgpt.com "Deconstructing the Mapper algorithm to extract richer ..."))
> ![[Pasted image 20250922105852.png]]

---

## 5.5 네트워크 과학 — 확산/전염의 위상 지도

**Taylor et al. (2015) PNAS**: _Topological data analysis of contagion maps for examining spreading processes on networks_

- **데이터/문제**: 사회·생물 네트워크에서 **전염/확산**이 공간적 임베딩과 장거리 연결의 상호작용으로 어떻게 진행되는가.
    
- **방법**: 네트워크 위 확산 동역학(예: Watts threshold)을 시뮬레이션해 각 노드의 발화 시간 등으로 **contagion map**을 구성 → 이 점군의 **위상(차원/루프)** 을 PH로 분석 → 저차원 임베딩과 함께 파형(파동) vs 점프식 확산을 구분.
    
- **결과/의의**: **파동형 전파**와 **원거리 점프 전파**가 만들어내는 **위상 서명**이 뚜렷이 구분됨을 보임. 네트워크 지리성·항공망 등 현실적 장거리 간선이 확산 패턴의 위상을 어떻게 바꾸는지 정량화하는 틀 제공. 이후 파생 연구로 절단 맵/변형 맵까지 확장. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4566922/?utm_source=chatgpt.com "Topological data analysis of contagion maps for examining ..."))
  ![[Pasted image 20250922105924.png]]
    

---
## 요약

- **생물학**: 분자 수준의 형태–기능 연계를 **PH 지표**로 모델링. ([arXiv](https://arxiv.org/abs/1412.2779?utm_source=chatgpt.com "Persistent homology analysis of protein structure, flexibility and folding"))
    
- **의학(EEG/fMRI)**: **비정상 뇌 상태**를 위상 서명으로 검출/분류. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10773586/?utm_source=chatgpt.com "Automatic epileptic seizure detection based on persistent ..."))
    
- **금융**: 위기 전 **구조 전이**를 landscape 지표로 조기 탐지. ([arXiv](https://arxiv.org/abs/1703.04385?utm_source=chatgpt.com "Topological Data Analysis of Financial Time Series: Landscapes of Crashes"))
    
- **추천**: 다중 뷰 구조를 **위상적으로 융합**해 군집/추천 보강. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050918322336/pdf?md5=7d4874eb9ba70880e2bdbdd3baf6b159&pid=1-s2.0-S1877050918322336-main.pdf&utm_source=chatgpt.com "Topological multi-view clustering for collaborative filtering"))
    
- **네트워크**: 확산 패턴의 차이(파동 vs 점프)를 **위상 지도**로 구분. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4566922/?utm_source=chatgpt.com "Topological data analysis of contagion maps for examining ..."))
    
---
## 6. 장점과 한계

- **장점**
    
    - 노이즈에 강건 (stability theorem 존재)
        
    - 데이터의 전역적 형태를 포착
        
    - 데이터 유형 제약이 적음 (수치, 범주, 혼합형 모두 가능)
        
- **한계**
    
    - 계산 복잡도 매우 큼 ($O(n^2)$ 이상의 메모리/시간)
        
    - metric, filtration 방식 선택에 따라 결과 변동
        
    - 해석이 직관적이지 않아 도메인 전문가와 협업 필요
        

---

## 7. 결론

TDA는 **점군 데이터 → 단순복합체 → 필트레이션 → 호몰로지(quotient space) → 지속 호몰로지 → 벡터화**의 과정을 통해 데이터의 전역 구조를 정량적으로 분석하는 방법론이다.  
공공기관 프로젝트 맥락에서는 **데이터의 구조적 특징을 노이즈에 강건하게 요약**하는 도구로서, 금융·추천·시계열 데이터에 새로운 차원의 feature를 제공할 수 있다.

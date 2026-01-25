![[자료/RecSys/RecSys 기초 프로젝트/[RecSys 기초 프로젝트] (3강) Collaborative Filtering 1.pdf]]

# 정의
CF : 많은 유저들로부터 얻은 기호 정보를 이용해서 유저 관심사를 자동으로 예측하는 방법

# 분류
## Neighborhood-based

목적 : 유저 u가 평점을 어떻게 매길거냐
장점 : 구현이 쉬움, 이해도 쉬움
단점
Scalability(아이템이나 유저가 많아지면 취약해짐)
Sparsity(주어진 평점/선호도가 적으면 성능이 낮아짐)

이거 적용하려면 sparsity ratio가 99.5% 안넘는게 좋음

### User-based CF

### Item-based CF

이 유사도는 norm으로 top-k개 보는건가?
## K-NN CF
NBCF의 한계 : 유저가 많아지면 연산할건 많아지는데 성능은 떨어짐.
그 유저 풀을 좁히면 되잖아? 해서 여기서 클러스터링하는거네

### 유사도 알고가자 Similarity Measure
#### Mean Squaed Difference Sililarity
이거 다른데선 잘 안쓰고 추천시스템 분야에서만 사용함
유클리드 거리의 역수(+1)

####  Cosine Similarity

결국 내적 써서 각도 구하는거임

#### Pearson Similarity
벡터를 표본평균으로 정규화 한번 하고 코사인 유사도를 구함.(평점 후하게 주는 유저도 사용 가능하겠지 대충)


#### Jaccard Similarity
집합의 개념을 사용함. 거리가 아니라 봤냐안봤냐 씀. 이걸 근데 어떻게 점수 예측하는데 쓰지?
4점 이상만 비교한다던지 기준을 정해버리면 쓸수있음

## Rating

### Absolute Rating

### Relative Rating
Deviation 써서 정규화시켜버리는거


![[자료/RecSys/RecSys 기초 프로젝트/[RecSys 기초 프로젝트] (4강) Collaborative Filtering 2.pdf]]

## Model-based

유저 정보로는 부족하다. 데이터 자체의 패턴을 사용하자. 모델 돌려서 파라미터를 사용?

앞서 UBCF의 단점 대부분 개선시킴
### 장점
1. 모델을 학습시켜서 이걸 쓰는거라 유저데이터 다 쓰는거보다 훨 가벼옴
2. Sparsity / Scalability 문제 개선함
3. Overfitting 방지함
   이웃 기반은 주변 유저를 너무 따라가는데, 모델은 패턴 따라가는거라 훨 나을거임
4. LImited Coverage 극복함
   유저 기반은 유저 없으면 좀 힘든데, 이건 괜찮음

## Feedback

### Explicit Feedback
영화 평점, 레이팅 등 직접적으로 알 수 잇는 데이터

### Implicit Feedback
암시적인, 클릭 여부, 시청했나 등 간접적으로만 알 수 있는 데이터

## Latent Factor Model
유저와 아이템 관계를 잠재적 요인으로 표현


# SVD(Singular Vector Decomposition)

선대의 eigenvalue로 분해하는 ㅜ머 그런것들

### 한계점
분해하려는 행렬의 Knowledge 불완전하면 정의가 안됨
따라서, 결측된 entry를 전부 채워줘야함 -> 연산량 증가


# MF
User-Item의 Latent factor 행렬곱으로 분해하는 법

## 정규화

왜 하나? 확률적 경사하강법(SGD) 사용해서 rate 업데이트함.


### L1


### L2

## +$\alpha$

### Adding Biases
아이템이나 유저도 편향이 생길 수 있음. 그 편향을 구해줘서 조정하는거임.

### Adding Confidence Level
모든 평점이 신뢰도가 같지 않아서, 신뢰도 추가해주는거.(광고같이 특정 의도를 가진 평점이라거나, 테러하는 유저라던가..)

### Adding Temporal Dynamics
시간에 따라 변하는 유저, 아이템 특성 반영


# ML for Implicit Feedback

그래서 implicit Feedback을 머신러닝 모델에 어떻게 넣어야 하나?

## Alternative Least Square(ALS)
Sparse한 데이터에 대해 SGD보다 더 Robust함
대용량 데이터 병렬처리하여 빠른 학습 가능

유저 P와 아이템 Q를 번갈아가며 업데이트함.

### 목적함수

# Bayesian Personalized Ranking
사용자한테 순서가 있는 아이템 리스트 제공


## Maximum A Posterior(최대 사후 확률 추정)
평점 예측값과 실제 값의 차이를 시그모이드에 넣어서 다 곱해버림

## LEARNBPR
Bootstrap 기반의 SGD 사용함.
선호 데이터와 선호하지 않는 negative sampling 동일 비율로 섞어서 랜덤샘플링하는거인ㄷ긋?



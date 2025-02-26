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


## Hybrid
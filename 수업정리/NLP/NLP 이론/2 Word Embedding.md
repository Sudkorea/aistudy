![[[NLP 이론] (2강) Word Embedding (1).pdf]]

# One-Hot Encoding

단어를 Categorical variable로 Encoding한 벡터로 표현

내적 유사도는 항상 0, 유클리드 거리는 항상 $\sqrt{2}$

단점 : 이거 단어라고 치면, 너무 sparse할거야
그래서 이거 저장하는 법이, `[0 0 2.1 0 -4.7 0]` 보단 `{(3, 2.1), (5, -4.7)}` 이렇게 저장하면 메모리 꽤 줄임

# Distributed Vector Representation

## Distributed Vector(Dense vector)
단어를 0으로 나타내는거 말고, 뭔가 의미를 가진 실수로 나타내는거
유클리드 거리, 내적, 코사인 유사도는 단어간 의미론적 유사성을 나타냄

## Word2vec
워드들을 Dense vector로 표현하는 Word Embedding의 대표적인 방법론
주변 단어들의 정보들을 이용해 단어 벡터 표현
어떤 단어가 다른 단어랑 같이 나왔나? 이거 쓰는듯함. Cat의 단어 앞뒤로 뭐가 올까?

파라미터 
Window size : 단어 앞뒤의 단어 몇개까지 볼건지

Input laver- > hidden layer -> Output layer
이 알고리즘은 two-layer neural net으로 이루어짐.

Word2Vec이 뭔지 고찰 : 단어들을 어떤 vector space에다 mapping하는 기법임. Germany 벡터랑 Capital 벡터 합치면 Berlin 뜨도록 만드는건데, 이 상관관계 학습하는데 neural network가 쓰이는듯. 그 관계는 많은 문장들에서 skip gram으로 주변을 넣거나, 문맥 단어를 전부 넣어서, 단어들 중심에 뭐가 있는지 예측하거나, 문장에 구멍뚫린쪽에 뭐나올지 예측하는 쪽으로 흘러가는데, 그게 CBOW, Skip-gram으로 나타나는듯?

---
# **One-Hot Encoding**

단어를 **범주형 변수(Categorical Variable)**로 인코딩하여 벡터로 표현하는 방법.

- 단어의 개수(어휘 수, Vocabulary Size)만큼의 차원을 가지며, 특정 단어에 해당하는 인덱스만 `1`이고 나머지는 `0`으로 설정됨.
- 예를 들어, 어휘 크기가 5인 경우:
    - `"cat"` → `[1, 0, 0, 0, 0]`
    - `"dog"` → `[0, 1, 0, 0, 0]`

### **단점**

- **희소성(Sparsity):** 대부분의 값이 `0`이라 비효율적임.
- **유사도 측정 불가능:** 내적 유사도는 항상 `0`, 유클리드 거리는 항상 $\sqrt{2}$이므로 단어 간 유사도를 비교하기 어려움.

### **메모리 효율화**

- 벡터가 매우 희소하기 때문에 메모리 낭비를 방지하기 위해 **희소 표현(Sparse Representation)**을 사용함.
- 예를 들어, `[0, 0, 2.1, 0, -4.7, 0]`를 `{(3, 2.1), (5, -4.7)}`처럼 저장하면 메모리 사용량이 감소함.

---

# **Distributed Vector Representation (분산 벡터 표현)**

- 단어를 0과 1이 아닌, 의미를 반영한 **실수(Dense Vector)** 값으로 나타내는 방법.
- 단어 간 유사성을 비교할 수 있도록 **유클리드 거리(Euclidean Distance)**, **코사인 유사도(Cosine Similarity)** 등의 연산이 가능함.

## **Word2Vec**

- 대표적인 **Word Embedding 기법** 중 하나.
- **주변 단어(Context Words)**의 정보를 이용해 단어 벡터를 학습함.
- 즉, 어떤 단어가 **어떤 단어와 함께 등장하는지**를 바탕으로 벡터를 학습.

## **Word2Vec이 Neural Network를 사용하는 이유**

Word2Vec은 Neural Network를 활용하여 단어들 간의 관계를 학습
여기서 뉴럴 네트워크는 복잡한 비선형 구조가 아니라, 단순히 **입력과 출력 사이의 가중치 행렬을 학습하는 역할**

### **구조:**

- **입력층 (Input Layer):** One-hot encoding된 단어 벡터
- **은닉층 (Hidden Layer):** 학습할 **단어 임베딩 벡터** (`V × N` 행렬)
- **출력층 (Output Layer):** 목표 단어 예측 (Softmax 또는 Negative Sampling)

이렇게 보면, Word2Vec의 뉴럴 네트워크는 단순히 **단어의 연관성을 학습하는 가중치 행렬을 최적화하는 과정**

---

## **Word2Vec이 단어 관계를 학습하는 방법**

Word2Vec은 **단어 간의 상관관계를 학습하는 방식**으로 **CBOW**와 **Skip-gram**이 있다.

### **1) CBOW (Continuous Bag of Words)**

- **여러 개의 주변 단어(Context Words)를 보고, 중심 단어(Target Word)를 예측**하는 방식
    
- 즉, **문장에서 빠진 단어를 예측하는 방식**
    
- 예제:
    
    - "The **cat** is sitting on the **mat**."
    - 입력: `["The", "is", "sitting", "on", "the", "mat"]`
    - 예측할 중심 단어: `"cat"`
    
    → **CBOW는 문맥(Context) → 중심 단어(Target) 예측** 방식
    

### **2) Skip-gram**

- **중심 단어(Target Word)를 보고, 주변 단어(Context Words)를 예측**하는 방식
    
- 즉, **특정 단어를 기준으로 주변 단어들을 예측하는 방식**
    
- 예제:
    
    - 중심 단어: `"cat"`
    - 예측할 주변 단어: `["The", "is", "sitting", "on", "the", "mat"]`
    
    → **Skip-gram은 중심 단어(Target) → 문맥(Context) 예측** 방식

보통 Skip-Gram 방법이 성능은 더 좋은데, 그만큼 학습 시간이 더 오래걸림

---

보통 50~300 차원의 벡터공간 사용함.
Negative Sampling 사용, Hierachical softmax : 출력층의 계산 복잡도 줄이는 방법

### 한계점
- 동음이의어 구분 어려움
- 단어의 문맥적 의미변화를 못잡아냄
- 이 한계를 극복하려고 발전한게 BERT, ELMo
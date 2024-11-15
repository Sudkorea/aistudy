## 0. Abstract

Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the ‘context’ of users’ activities on the basis of actions they have performed recently. 

To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user’s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. 

Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. The goal of our work is to balance these two goals, by proposing a self-attention based sequential model (SASRec) that allows us to capture long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC). 

At each time step, SASRec seeks to identify which items are ‘relevant’ from a user’s action history, and use them to predict the next item. Extensive empirical studies show that our method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models. Visualizations on attention weights also show how our model adaptively handles datasets with various density, and uncovers meaningful patterns in activity sequences.


여기선 딱히 어려운 용어는 없어서, 용어정리는 안하고 감.

## 내가 이해한 바로는
Sequential dynamics는 현대 추천 시스템에서 유저의 'context data'를 반영하기 위해 자주 쓰이는 방법이다. 이를 위해 두 가지 방법을 사용하는데, MC, RNN이다. MC는 일반적으로 sparse한 데이터에서 성능이 좋게 나오고, 추론을 하기 위해 최근 몇개의 데이터를 사용한다. RNN은 더 긴 시간의 데이터를 사용하고, 좀 더 dense한 데이터에서 성능이 좋게 나온다. SASRec의 목표는, 이 두가지 모델의 한계점을 극복하기 위해 transformer의 self-attention 메커니즘을 도입하여, RNN과 MC의 장점을 섞은 모델을 만드는 것이다. 이는 sparse한 데이터에도, dense한 데이터에도 CNN/RNN/MC 기반의 모델보다 성능이 좋게 나온다.

# 1. Introduction

여기는 이제 abstract에서 말한 것들을 조금 더 구체적으로 이야기하고 있음. 마르코프 체인과 RNN에 대해 더 설명한 뒤, Transformer가 발표된 이후 거기에 영감을 받아, self-attention을 가져와 SASRec를 구현한다고 함.

# 2. Related Work
2장 related work는, 말그대로 관련 연구를 소개하고 있음.
## a. General Recommendation
일반적인 추천 시스템 유저 피드백은 명시적 또는 암시적으로 나뉘는데, 모델의 암시적 피드백은 명확히 관찰된 것은 아니라 모델링에 사용하기 까다로움.(명확히 관찰되지 않았다는 것은, 클릭했다는 정보로 살까 말까를 판단할 수 없기 때문) 

이 문제를 해결하기 위해 사용하는 두 방법이 pairwise와 pointwise임. 

(잠시 내 이해를 위해 두 방법에 대해 짚고 넘어가자면, pointwise는 한 유저가 한 아이템을 다룰 때 어떻게 평가하는가에 초점을 맞추는 것이고, pairwise는 여러 가지 아이템 중 어떤 것들을 순서대로 선호할 것인지에 초점을 맞추는 방법론임.) 

그 이후, 유저와 아이템의 인접행렬을 다루는 MF method에 대해 다루고, 아이템 간의 유사도를 다루는 방식인 ISM에 대해 다룸. 

최근에는 deep learning이 추천 시스템에 많이 사용되는 추세임. 
두 가지 큰 줄기에서, context를 다루는 pairwise 기반의 모델과, MF를 개선한 방향인 pointwise 기반의 MLP, AutoRec 같은 딥러닝 구조를 사용함.

## b. Temporal Recommendation
이 거지같은 키보드를 당장 부수던지 해야지. mac 세팅이랑 window 세팅이랑 계속 왔다갔다해서, 한영키(alt)를 눌렀는데 window가 튀어나온다. 차라리 이게 계속 이러면 그냥 바뀐 키에 익숙해지면 되는데, 자꾸 왔다갔다해서 매우 화가난다. epomaker 키보드는 생긴건 이쁜데 가격 생각하면 퀄이 거지같다.

Netflix prize로 거슬러 올라가면, 임시적인 추천은 timestamp of users activites 기반으로 추천하는데 강점이 있음. TimeSVD++는 시간의 흐름에 따라 분석하는 모델에서 필수적임. 
(이제 우리가 할) Sequential recommendation은 이 방법과 약간 차이가 있는 것이, 수열을 분석하는데 있어 시간변수를 좀 독립적인 어떤 변수로 다룬다는 차이가 있음.

이 부분은 내가 이해가 약해서, gpt 답변도 달아둠.

### GPT 답변
정확하게 이해하셨습니다! Temporal Recommendation과 Sequential Recommendation 간의 차이를 잘 파악하셨네요. 여기서 몇 가지 보충할 점을 정리해드릴게요.

1. **Temporal Recommendation**: 이 접근법은 사용자의 행동이 시간에 따라 변한다는 점에 주목합니다. Netflix Prize에서도 확인되었듯, 추천 시스템이 특정 시간에 발생한 사용자 활동을 기반으로 추천할 때 더 높은 정확도를 보입니다. TimeSVD++는 시간에 따른 변화가 어떻게 추천에 영향을 미치는지를 모델링하는 데 유용하며, 특정 시간에서 사용자와 아이템 간의 관계를 가중치로 조정하는 방식입니다.
    
2. **Sequential Recommendation과의 차이**:
    
    - Temporal Recommendation이 시간의 흐름에 따라 사용자 선호도가 어떻게 변하는지 파악하는 데 집중한다면, Sequential Recommendation은 특정 행동이 일어나는 순서에 더 중점을 둡니다.
    - Sequential Recommendation에서 시간은 단순히 연속된 행동의 순서에서 파생되는 변수로 취급됩니다. 즉, 시간 변수는 수열의 순서 정보로 대체되며, 실제 시간 간격(예: 몇 시간 후에 일어난 행동인지)은 반드시 중요하지 않다고 여깁니다.

요약하자면, Temporal Recommendation은 행동이 발생한 정확한 시간을 고려하여 선호도를 예측하는 데 주력하고, Sequential Recommendation은 사용자가 행동한 순서나 패턴을 주로 분석하며 시간 간격 자체보다는 순차적인 관계를 더 중시합니다. SASRec과 같은 Sequential Recommendation 모델은 이 순차적 관계를 독립적인 변수처럼 취급하여 시간에 덜 의존하는 특성을 갖는 것이 특징입니다.


## c. Sequential Recommendation
이 부분이 sequence라는 개념을 어디서 갖다쓴건지 저번 강의 들을때 헷갈렸었음. 아이템을 통해 다음 아이템을 계속해서 추천하는 것이 sequence a_1, a_2, ... 와 같은 귀납적 정의와 닮아있어서 그럼)

For instance, FPMC fuses an MF term and an item-item transition term to capture long-term preferences and short-term transitions respectively (여기서 "fuses"라는 말은 두 가지 요소를 결합한다는 의미.) FPMC는 사용자의 장기 선호도를 표현하는 MF(Matrix Factorization) 요소와, 아이템 간 전환을 나타내는 item-item transition 요소를 결합하여, 장기적인 선호도와 단기적인 전환을 모두 포착하는 모델임. 사용자가 오랜 시간 좋아했던 아이템의 특성을 MF에, 최근에 상호작용한 아이템 특성(item-item transition)을 고려하여 다음 추천을 만듦.

당연하게도, 이전 내용을 통해 다음 내용을 생성하는 개념을 다뤄야 하므로 마르코프 체인은 필수임. 1차 마르코프 체인은 바로 직전 데이터로 다음 데이터를 만드는데 효과적이며, 당연하게도 고차원 마르코프 체인일수록 그 이전 데이터도 사용하는 느낌임. 

구체적으로 이 마르코프 체인을 사용하는건 CNN-based model임. 이 부분은 이해가 어려웠음.

- Convolutional Sequence Embedding 모델은 embedding matrix $L$을 사용하여, 이전 행동을 embedding으로 보고 합성곱을 수행하는 방식입니다. 여기서 convolutional operations를 통해 최근 행동들과의 관계를 파악하면서 sequence에서 중요한 패턴을 잡아내죠.
- 합성곱(convolution)은 이전 행동들의 embedding을 결합하여, 특정 패턴을 강조하거나 반복적 관계를 학습하는 데 유리한 특징을 가집니다. convolutional operations에 대해 더 깊이 파고들면, 이미지 처리와 비슷한 방식으로 연속된 행동을 다룬다고 생각하면 됩니다.

MC를 사용하는 다른 갈래는 RNN에 이식한 형태임. GRU4Rec는 클릭 데이터를 다음 추천에 활용하고, Top-N을 통해 학습함. 이를 통해 RNN이 저번 데이터와 이번 데이터만 학습하게 만들어서, 효율성을 올림

## d. Attention Mechanisms 
어텐션 매커니즘은 이미지, 기계번역 등 여러 곳에서 쓰임. 이 아이디어의 핵심은 역시 입력 데이터에서 연관있는 부분에 집중하여 sequential output을 생성하는 것임. 
어텐션 기반 모델의 또다른 장점은, 모델이 주로 해석 가능하다는 점임. 최근에 어텐션이 추천 시스템에도 사용되는 추세인데, AFM같은 예시가 있음. 

(작성자가 Transformers 많이 좋아하나봄. state-of-the-art라는 표현을 두번이나 썼네. ai 공부하는 사람중 안그런 사람 어딨겠냐만은..) 

어텐션이라는 개념은 예전엔 주로 다른 모델에 섞어서 쓰였는데, 순수 어텐션 모델인 transformer는 기계번역 분야에서 예술의 경지에 오름. 이 트랜스포머에 영감을 받아서, 어텐션으로 이루어진 모델을 만들고자 함.

와 나 여기까지 어떻게 번역기 안쓰고 읽었냐
2시간 후딱갔네

# 3. METHODOLOGY
여기 LaTEX 쓸일이 너무 많아서, 내가 정리하고, gpt가 수식이랑 문장 다듬어준 버전으로 올려놓음.

## Notation

|                            Notation                            | Description                                                                                                                                                     |
| :------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                   $\mathcal{U}, \mathcal{I}$                   | 유저, 아이템셋                                                                                                                                                        |
|                       $\mathcal{S}^{u}$                        | 유저 u에 대한 historical interaction sequence. <br>이제 이걸 시간순으로 나타내면 $(\mathcal{S}^{u}_{1}, \mathcal{S}^{u}_{2}, ..., \mathcal{S}^{u}_{\vert \mathcal{S}^{u} \vert})$ |
|                       $d \in \mathbb{N}$                       | 임베딩벡터 차원                                                                                                                                                        |
|                       $n \in \mathbb{N}$                       | 최대 수열 길이                                                                                                                                                        |
|                       $b \in \mathbb{N}$                       | self-attention block 개수                                                                                                                                         |
| $\mathsf{M} \in \mathbb{R}^{\vert \mathcal{I}\vert \times d }$ | item enbedding matrix                                                                                                                                           |
|           $\mathsf{P} \in \mathbb{R}^{n \times d }$            | positional embedding matrix                                                                                                                                     |
|        $\hat{\mathsf{E}} \in \mathbb{R}^{n \times d }$         | input embedding matrix                                                                                                                                          |
|        $\mathsf{S} ^{(b)} \in \mathbb{R}^{n \times d }$        | b번째 self-attention layer 이후 아이템 임베딩                                                                                                                             |
|       $\mathsf{F} ^{(b)}  \in \mathbb{R}^{n \times d }$        | b번째 feed-forward network 이후 embedding network                                                                                                                   |
이건 직접 만듦
### GPT 답변(P, F^b 이해가 안가서 정리해둠.)

#### **1. Positional Embedding Matrix ($\mathsf{P}$)**

##### **역할**
- **Transformer 구조**에서 sequence 데이터의 순서를 표현하기 위해 도입된 개념입니다.
- Self-attention 메커니즘은 각 위치의 데이터 간 관계를 계산하지만, **위치 정보**(순서)를 내재적으로 고려하지는 않습니다. 이를 해결하기 위해 각 아이템의 위치 정보를 나타내는 **positional embedding**을 추가합니다.

##### **SASRec에서의 활용**
- $\mathsf{P} \in \mathbb{R}^{n \times d}$: 길이 $n$의 sequence에서 각 위치(1번째, 2번째, ..., $n$번째)에 대한 $d$차원의 embedding을 저장합니다.
- $\hat{\mathsf{E}} = \mathsf{E} + \mathsf{P}$ 형태로, positional embedding을 input embedding에 더해줍니다.
  - 여기서 $\mathsf{E}$는 item embedding matrix에서 나온 값입니다.
- 결과적으로, $\mathsf{P}$는 아이템 간 상대적인 순서를 모델에 알려주는 역할을 합니다.

##### **직관적인 이해**
예를 들어, 아이템 $i_1$, $i_2$, $i_3$의 embedding은 같을 수 있지만, $i_1$이 sequence의 첫 번째 위치에 있는지, 두 번째 위치에 있는지를 $\mathsf{P}$가 알려줍니다.

---

#### **2. Feed-Forward Network ($\mathsf{F}$)**

##### **역할**
- Self-attention layer를 통과한 아이템 embedding $\mathsf{S}^{(b)}$에 대해, 각 아이템의 embedding을 더 복잡한 표현으로 변환합니다.
- 이 과정은 **비선형 변환**을 통해 embedding의 표현력을 높이는 역할을 합니다.

##### **구조**
Feed-forward network는 주로 다음 두 단계로 구성됩니다:
1. 선형 변환 (Fully Connected Layer): $\mathbb{R}^d \to \mathbb{R}^{d_{\text{hidden}}}$
   - 아이템 embedding을 더 높은 차원으로 변환합니다.
2. 활성화 함수 (ReLU 등): 비선형성을 추가해 학습 능력을 강화합니다.
3. 다시 선형 변환: $\mathbb{R}^{d_{\text{hidden}}} \to \mathbb{R}^d$
   - embedding의 차원을 원래대로 되돌립니다.

##### **SASRec에서의 활용**
- $\mathsf{F}^{(b)} = \text{FFN}(\mathsf{S}^{(b)})$
  - self-attention의 출력을 입력으로 받아, 각 아이템 embedding을 독립적으로 변환합니다.
- self-attention은 관계를 학습하지만, feed-forward network는 개별 아이템의 표현을 풍부하게 만드는 역할을 합니다.

#### **직관적인 이해**
예를 들어, self-attention에서 "어떤 아이템이 중요하다"는 관계 정보를 학습했다면, feed-forward network는 이 정보를 바탕으로 해당 아이템 embedding을 더 복잡하고 유용한 형태로 바꿉니다.

---

#### 요약
- **Positional Embedding Matrix**는 sequence의 순서 정보를 모델에 전달하기 위해 사용됩니다.
- **Feed-Forward Network**는 self-attention 이후 각 아이템 embedding을 더 정교한 표현으로 변환하는 역할을 합니다.

## a. Embedding Layer
Training sequence $\mathcal{S}^{u}$를 고정 길이 $n$의 수열로 변환할 때, $\mathcal{S}^{u} = (s_1, s_2, \dots, s_n)$로 설정된다. 데이터가 $n$을 초과하면 최근 $n$개 항목만 유지해 sequence 길이를 제한함. 각 $s_i$에 대해 아이템 embedding vector가 생성되며, positional embedding matrix $\mathsf{P}$가 사용되어 위치 정보를 추가함.

이때 embedding layer의 전체 출력은

$$
\hat{\mathsf{E}} = \mathsf{E} + \mathsf{P}
$$

다시 정리하면,
- $\mathsf{E} \in \mathbb{R}^{n \times d}$는 아이템 embedding matrix에서 생성된 아이템의 embedding
- $\mathsf{P} \in \mathbb{R}^{n \times d}$는 positional embedding matrix로, 각 위치의 고유한 위치 정보를 제공

따라서,
$$
\hat{\mathsf{E}} = [\mathsf{M}_{s_1} + \mathsf{P}_1; \mathsf{M}_{s_2} + \mathsf{P}_2; \dots; \mathsf{M}_{s_n} + \mathsf{P}_n]
$$

이 식은 각 위치 $i$에서 아이템 embedding $\mathsf{M}_{s_i}$와 위치 embedding $\mathsf{P}_i$를 더해 sequence의 최종 embedding $\hat{\mathsf{E}}$을 형성함.

논문 작성자가 fixed positional embedding을 시도했으나, 실험 결과 dynamic positional embedding(위치마다 학습된 embedding)을 사용하는 것이 더 나은 성능을 보여준다고 보고했음. 이는 SASRec에서 sequence 내에서 각 위치에 맞는 특성을 학습하는 데 도움이 되었다고 볼 수 있음.

## b. Self-Attention Block

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right)V
$$

- $Q$ (Query), $K$ (Key), $V$ (Value)는 input embedding $\hat{\mathsf{E}}$를 변환한 행렬
- $\frac{Q K^\top}{\sqrt{d}}$: Query와 Key 간의 유사도를 측정하고, $\sqrt{d}$로 나눠 계산의 스케일을 안정화시킴
- $\text{softmax}$: 유사도를 확률로 변환하여 각 Query가 Key에 얼마나 집중할지 결정함
- $V$: Value는 실제로 모델이 집중할 정보의 내용을 담음

### Self-Attention Layer
NLP에서는 보통 $K = V$로 설정함. 이 논문에서도 이 점을 따르며, SASRec에서는 $\hat{\mathsf{E}}$를 input으로 사용하여 self-attention을 정의함.

$$
\mathsf{S} = \text{SA}(\hat{\mathsf{E}}) = \text{Attention}(Q, K, V)
$$

여기서:
- $Q = \hat{\mathsf{E}} \mathsf{W}^Q$, $K = \hat{\mathsf{E}} \mathsf{W}^K$, $V = \hat{\mathsf{E}} \mathsf{W}^V$
- $\mathsf{W}^Q$, $\mathsf{W}^K$, $\mathsf{W}^V$는 각각 $d \times d$ 크기의 학습 가능한 행렬로, 모델이 sequence 내의 관계를 유연하게 학습하도록 도움.

### Causality

Self-attention에서 각 아이템 embedding $\mathsf{S}_t$는 모든 이전 아이템뿐만 아니라, 이후 아이템(subsequent items)의 정보까지 반영하게 됨. 하지만 SASRec의 목표는 **t+1번째 아이템을 예측하기 위해 t번째까지의 정보만 사용하는 것**이므로, 이후 아이템 정보를 포함하는 것은 문제를 잘못 정의하는 것이 됨.

이를 해결하기 위해 논문에서는 **causality constraint**를 도입하는데, 이 제약은 Query ($Q_i$)와 Key ($K_i$) 간 연결을 제한해, 시간순으로 미래의 정보가 과거에 영향을 미치지 못하게 함.

이를 구현하기 위해 **masking**을 적용하여, $t$ 시점 이전 정보에만 접근할 수 있도록 하는데,

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \mathsf{M}\right)V
$$

- $\mathsf{M}$은 **masking matrix**로, 이후 시점의 정보를 차단하는 역할을 한다. 예를 들어, $M_{ij} = -\infty$로 설정하면, softmax 계산에서 해당 위치의 값이 0으로 설정됨.
- 이로 인해 모델은 과거 정보에만 집중하며, 미래 정보를 포함하지 않도록 보장됨.


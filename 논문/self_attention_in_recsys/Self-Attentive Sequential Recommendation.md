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

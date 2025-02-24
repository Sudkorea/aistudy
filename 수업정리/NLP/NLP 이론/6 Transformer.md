![[[NLP 이론] (7강) Transformer 1.pdf]]

# Self-Attention 개념

Seq2Seq + Attention은 Encoder, Decoder, Attention 3개 파트로 나뉨

RNN 떠올려보자. x_1에서 h_3까지 가려면 2단계 거쳐야하는데, 그냥 바로 갈 순 없나?(거치다보면 vanishing gradient 일어나잖아) -> Long term dependency 문제를 해결하기 위해 사용하는게 Attention

Hash Table T를 가정함.

Query 입력 -> 정확하게 일치하는 key의 Value를 출력함.(dict 생각해봐)

QKV 이렇게 생각하니까 되게 쉽네

-> Attention은 Soft matching을 통한 Hash Table
- Query와 정확히 일치하는 key가 없어도 유사한거 찾아서 활용함.
	- Query, Key, Value가 모두 벡터로 표현됨
	- Query-Key간의 Soft한 유사도를 바탕으로 Value의 가중 평균 계산


Hash Table T를 구성하는 방법

1 Key-Value값을 input x_k, x_k 쌍으로 만든다
2 Softmax를 통해 코사인 유사도 구한걸 정규화시켜서 Key로 사용한다(?)
3 x_1은 Query이자 Key이자 Value였는데?? 

자기 자신을 Query로 사용하여 필요한 정보를 끌어오는게 Self-Attention임.

Q, K, V에 각자 선형변환 적용함

#### 이게 대체 무슨소리에요?

Q
```
그럼, self_attention은 Q,K,V 전부를 담당하는 한 문장의 각 단어들로 이루어진 X에 대해서, 각 용도에 맞는 Linear Transform을 적용시켜 dim이랑 값을 어떻게 잘 만져준 다음에, Q와 K의 Transpose를 행렬곱해서(이는 결국 각 query에 대해 '모든' Key와의 연관성을 연산해서 value를 구하기 위함), 그 값을 softmax써서 정규화시켜주면, 각 문장의 위치에 어떻게 집중해야 할지 가중치가 구해지는거고, 이를 Value에 곱해줘서, Value 저장해둔것 중 어떤 부분에 집중해서 값을 뽑아내는 연산을 진행하는거고, 이거로 output을 만들어내는 과정

이렇게 이해하면됨? 

한단계 더 나가면, self-attnetion을 학습시키는 건 결국 선형 변환 X를 어떻게 잘 만져주는 작업이라고 보면 됨?
```

A
```
네, 정확히 이해하셨습니다! 아주 명확하게 정리해주셨네요.

Self-attention의 핵심을 다시 한번 확인해보면:

1. 입력 처리:
- 하나의 입력 X가 Q, K, V 세 가지 역할을 모두 수행
- 각 역할에 맞는 가중치 행렬(W_q, W_k, W_v)과 곱해서 변환
- 이 가중치 행렬들이 바로 학습 대상

2. Attention 계산:

	Q @ K.transpose() 의미:
	- 각 위치(Query)에서 모든 위치(Key)를 참조
	- 예: "나는 사과를 먹었다"에서
	  "사과"(Query)와 모든 단어(Key)의 연관성 계산


3. 학습 관점:
- 네, 정확합니다! Self-attention 학습의 본질은 W_q, W_k, W_v 행렬을 최적화하는 것
- 이 가중치들이 어떻게 조정되느냐에 따라:
  - 어떤 단어들끼리 강하게 연결될지
  - 어떤 문맥 정보가 중요하게 여겨질지
  - 각 단어의 의미 표현이 어떻게 될지가 결정됨

결국 이 메커니즘으로 문장의 각 부분이 서로를 얼마나 "주목"해야 하는지를 모델이 학습하게 되는 것이죠.

```
# Scaled dot-product attention 계산과 원리의 이해

Self-Attention에서, d_k가 커지면 Dot-product q,k의 분산이 증가함
-> Softmax 내 특정 값이 유난히 커질 수 있음
-> Softmax 출력이 한 값에 편중됨
-> Gradient가 매우 작아짐(이거때매 만든건데?)

해결법
Query/Key 벡터 길이만큼 출력값의 분산값을 줄임

그니까, Q@K 하고 sqrt(d_k) 한번 나눠준 다음 value에다가 적용하는거같음

# Multi-head attention의 필요성과 적용법

지금까지 배운거에서 문제점 : 결국 확률변수라는게 하나의 Softmax 기반한 결과값이 나옴. 그니까, I became Chosun Swordmaster -> became에 너무 어그로가 쏠림. 

이래서 Mutlihead attention을 쓰는거임. 여러 attention 써서 이거저거 보려고

누군 문맥 보고, 누군 문법상 중요한거 보고 이런거임


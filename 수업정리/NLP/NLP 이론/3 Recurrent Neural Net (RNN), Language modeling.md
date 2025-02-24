
![[[NLP 이론] (3강) RNN과 Language Modeling.pdf]]

# RNN
가변적 길이의 Sequence 자료를 입력으로 받아서 출력함.

$$h_t = f_{\theta}(h_{t-1}, x_t)$$
- $h_t$ : new state
- $f_{\theta}$ : parameter $\theta$를 가지는 어떤 function
- $h_{t-1}$ : old state
- $x_t$ : Time Step t에서의 입력 vector
- **!주의** : 모든 time step에서 같은 함수 $f_\theta$와 같은 Parameter $\theta$ 적용.

- Multi-layer RNN
  각 time stemp에서 input -> layer 1 ->layer 2 -> output
- Bidirectional RNN
  layer 1, 2를 forward / backward layer으로 사용함. 각자 input을 넣고, forward는 의미가 오른쪽으로 전이되고, backward는 왼쪽으로 전이됨. concat해서 이어붙여서 output 만듦

이거 말고도, 형태야 다양하게 가능함. 


문자열 단위 Language Model

- 각 문자를 One-hot encoding으로 입력
- 입력 $x_t$으로 Hidden state $h_t$ 계산
- Hidden state에서 출력 $y_t$ 계산
- Softmax를 통해 각 문자의 확률 계산
- Cross-entropy loss 계산 및 학습

이걸 Auto-Regressive Model으로 부름

sos eos(start/end of sentense)


Backpropagation Through Time(BPTT)


Truncated Bacpropagation through time(TBPTT)

![[[NLP 이론] (4강) Exploding and Vanishing Gradient of Recurrent Neural Networks.pdf]]
## [Exploding / Vanishing] Gradient

긴 문장 입력의 문제점 : 기울기 소실. 거리가 너무 멀면 저장이 끝까지 안간다

단순히 다항함수라고 생각해도, 초반 정보는 x^n 정도 먹을거고 곱하다보면 자연스레 작아질거임

이거 해결한게 Attention이랑 MAMBA(State Space Model)계열 아니였나

Gradient Clipping : gradient exploding 막으려고 최댓값 임의로 설정해주는거

이걸 해결하려고 나오는게 LSTM. 이건 다음 시간에~

---
# **Recurrent Neural Network (RNN)**

RNN은 **가변 길이의 시퀀스 데이터**를 입력으로 받아서 **순차적으로 처리하고, 이전 정보(Context)를 기억하여 다음 출력을 생성**하는 구조야.

## **1. RNN의 수식**

RNN은 **이전 상태($h_{t-1}$)와 현재 입력($x_t$)을 이용해 새로운 상태($h_t$)를 계산하는 함수**로 표현돼.

ht=fθ(ht−1,xt)h_t = f_{\theta}(h_{t-1}, x_t)

- $h_t$ : 현재 시점의 hidden state (메모리 역할)
- $h_{t-1}$ : 이전 시점의 hidden state
- $x_t$ : 현재 시점의 입력 벡터
- $f_{\theta}$ : 네트워크의 학습 가능한 함수 (파라미터 $\theta$ 포함)
- **모든 time step에서 동일한 함수 $f_\theta$와 동일한 파라미터 $\theta$ 사용**

즉, RNN은 **하나의 뉴런이 시간에 따라 반복적으로 사용되는 구조**이기 때문에 **가중치 공유(weight sharing)**가 발생해.

---

## **2. RNN의 다양한 구조**

### **(1) Multi-layer RNN**

- 여러 개의 RNN 층을 쌓아 깊이를 증가시키는 방식.
- 각 time step에서:
    - **입력** → **Layer 1** → **Layer 2** → **출력**
- 층을 깊게 만들면 **더 복잡한 패턴 학습 가능**.

### **(2) Bidirectional RNN (BiRNN)**

- RNN은 기본적으로 **과거 정보 → 현재 정보**의 단방향 구조인데, BiRNN은 **양방향으로 정보를 전달**해.
- 두 개의 RNN을 사용:
    - **Forward RNN:** 왼쪽 → 오른쪽으로 정보 전파
    - **Backward RNN:** 오른쪽 → 왼쪽으로 정보 전파
- 두 방향에서 계산된 hidden state를 **concat(연결)** 해서 최종 출력 생성.
- **예시:** 문장의 문맥을 더 잘 이해할 수 있도록 사용됨 (e.g., 기계 번역, 문장 태깅).

### **(3) 변형 가능**

- 기본 RNN 외에도 다양한 변형이 가능함 (e.g., LSTM, GRU).

---

## **3. RNN을 이용한 문자열 단위 Language Model**

- 입력을 **One-hot encoding**하여 사용.
- **각 문자($x_t$)를 입력받아 hidden state($h_t$)를 계산**.
- Hidden state에서 **출력($y_t$)을 Softmax를 통해 확률 분포로 변환**.
- **Cross-entropy loss**를 계산하여 학습.

이 모델은 **Auto-Regressive Model**이라고 불려.

- 이유? **이전 출력을 이용해 다음 출력을 예측하는 방식이기 때문**.

**(예시)**  
`"hello"`를 예측하는 경우:

1. `"h"`를 입력하면 `"e"`가 출력되도록 학습.
2. `"e"`를 입력하면 `"l"`이 출력되도록 학습.
3. 이런 방식으로 순차적으로 학습.

### **시작/종료 토큰 (SOS/EOS)**

- 문장 시작을 의미하는 `SOS(start of sentence)`, 문장 종료를 의미하는 `EOS(end of sentence)` 토큰을 추가.
- 문장의 시작과 끝을 명확하게 구분하여 모델이 문장을 생성할 수 있도록 함.

---

## **4. Backpropagation Through Time (BPTT)**

- RNN에서 **시퀀스 데이터에 대한 그래디언트(Gradient)를 계산하여 가중치를 업데이트하는 과정**.
- **기본 원리:** 일반적인 신경망처럼 **역전파(Backpropagation)**를 수행하되, **시간(Time) 방향으로 펼쳐진 RNN의 그래디언트를 역전파**하는 방식.

### **(문제점) Vanishing Gradient Problem**

- 시퀀스가 길어질수록, 초반 시점의 정보가 뒷부분으로 전파되지 않음.
- 즉, **초기 입력의 영향력이 점점 작아지는(Gradient Vanishing) 현상**이 발생.

---

## **5. Truncated Backpropagation Through Time (TBPTT)**

- **BPTT의 한계를 해결하기 위해 긴 시퀀스를 여러 개의 짧은 구간으로 나눠 학습**하는 기법.
- 너무 긴 시퀀스를 학습할 경우 **메모리와 연산량이 너무 커지는 문제**가 발생하므로, **고정된 길이로 잘라서(Truncated) 학습**함.

### **(BPTT vs. TBPTT)**

||BPTT|TBPTT|
|---|---|---|
|**방식**|전체 시퀀스를 한 번에 역전파|일정 길이로 나누어 학습|
|**장점**|전체 문맥을 반영 가능|메모리 효율적, 연산량 감소|
|**단점**|긴 시퀀스에서는 학습이 어려움|문맥 손실 가능성 있음|

---

## **RNN 정리**

|항목|설명|
|---|---|
|**입력/출력**|가변 길이의 시퀀스 입력 → 시퀀스 출력|
|**Hidden State**|이전 state와 현재 입력을 활용해 새로운 state를 계산|
|**Multi-layer RNN**|여러 개의 RNN 층을 쌓아 학습 성능 향상|
|**Bidirectional RNN**|양방향으로 문맥 정보를 활용하여 더 정교한 결과 도출|
|**Auto-Regressive Model**|이전 출력 값을 사용하여 다음 값을 예측하는 구조|
|**BPTT**|시간 방향으로 역전파하는 학습 방식|
|**TBPTT**|긴 시퀀스를 잘라서 학습하여 메모리 문제 해결|

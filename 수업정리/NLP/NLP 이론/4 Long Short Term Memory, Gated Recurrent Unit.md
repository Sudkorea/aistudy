![[[NLP 이론] (5강) LSTM과 GRU.pdf]]

# LSTM
RNN처럼 매 time step마다 동일 모듈을 적용함
LSTM 안에는 4개의 Vector를 계산하는 신경망 존재

인간의 단기기억능력 short term을 억지로 늘려놓은 꼴

### ResNet에서 LSTM까지

y = x + f(x)
지루함, 현학적임.
뭣
문제는 이게 그래디언트 소실남

y = x + (1-a)x + af(x)

이걸 조금 더 확장시키면

y = (1-a(x))x + a(x)f(x)
여기서 a(x) = sgm(g(x)) (sigmoid function)

가운데껀 가중치를 상수로 둔거, 아래껀 가중치를 원소별로 다르게 만든거임. 이걸 Highway Network라 부름.
이걸 조금 더 유연하게 확장시키면

y = b(x)x + a(x)f(x)

### LSTM의 구조

#### Notation
- f : forget gate, Cell의 내용을 지울지 결정함
- i : input gate, Cell의 내용을 적을지 결정함
- g : function output, Cell에 적을 내용
- o : Output gate, Cell의 내용을 출력할지 결정함
- W : Weight Matrix
- h : Hidden state
- c : Cell State
- x : input

그건 그렇고, 하이퍼볼릭tan 많이쓰네

GRU

LSTM의 Cell state랑 Hidden State를 하나의 Vector로 통합한거임. 경량화 버전

---

# **LSTM (Long Short-Term Memory)**

LSTM은 **RNN의 단기 기억 한계를 극복하기 위해 설계된 모델**이야.  
기존 RNN은 시퀀스 길이가 길어질수록 **기억 유지가 어렵고, 그래디언트 소실(Vanishing Gradient)** 문제가 발생하는데,  
LSTM은 **Cell State**를 추가하여 정보가 더 오랫동안 유지될 수 있도록 했어.

- **RNN처럼** 매 time step마다 동일한 연산 모듈을 반복적으로 적용
- **단기 기억(Short-term memory)을 강제로 연장**한 구조
- 내부적으로 **4개의 벡터 연산(게이트 연산)이 포함됨**

---

## **1. LSTM의 등장 배경**

### **1) RNN의 문제점 - Vanishing Gradient**

일반 RNN에서는 **오랜 시간 전의 정보가 사라지는 문제**가 있어.

- 예를 들어, 문장에서 `"The cat sat on the"`라는 문맥이 있을 때, `"mat"`을 예측해야 하지만,  
    **문장이 길어지면 앞쪽 정보("The cat")가 모델에 잘 전달되지 않음.**
- 이는 **그래디언트 소실(Vanishing Gradient) 문제** 때문임.

### **2) Residual Connection과 Highway Network에서 LSTM까지**

RNN의 문제를 해결하려는 여러 접근이 있었어.

#### **(a) Residual Connection (ResNet)**

y=x+f(x)y = x + f(x)

- 입력 $x$를 그대로 전달하면서 변화량 $f(x)$를 더하는 방식.
- 그래디언트가 사라지는 문제를 어느 정도 해결 가능.

#### **(b) Highway Network**

y=(1−a(x))x+a(x)f(x)y = (1 - a(x))x + a(x)f(x)

- 여기서 $a(x)$는 **sigmoid activation**을 이용해 입력이 얼마나 유지될지를 조절.
- 특정 뉴런이 중요한 정보를 **그대로 전달할 수 있도록 가중치를 조정**.
- → **LSTM도 같은 원리를 차용하여 정보가 오래 유지되도록 설계됨!**

---

## **2. LSTM의 구조**

### **기본 개념**

LSTM은 RNN과 달리 **Cell State($c_t$)**를 추가하여 중요한 정보를 오랫동안 저장할 수 있음.  
각 시점마다 4가지 벡터 연산(게이트 연산)이 수행됨.

### **Notation (기호)**

- **$c_t$** : Cell State (장기 기억 저장소)
- **$h_t$** : Hidden State (출력 및 단기 기억)
- **$x_t$** : 입력 벡터
- **$W$** : Weight Matrix
- **$f_t$** : Forget Gate
- **$i_t$** : Input Gate
- **$g_t$** : Cell Update Candidate
- **$o_t$** : Output Gate

### **LSTM Cell의 연산**

1. **Forget Gate ($f_t$)**  
    이전의 기억을 얼마나 유지할지 결정.
    
    ft=σ(Wf⋅[ht−1,xt]+bf)f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
    - 값이 `1`이면 기억 유지, `0`이면 삭제.
2. **Input Gate ($i_t$) & Candidate Cell Update ($g_t$)**  
    새로운 정보를 얼마나 받아들일지 결정.
    
    it=σ(Wi⋅[ht−1,xt]+bi)i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) gt=tanh⁡(Wg⋅[ht−1,xt]+bg)g_t = \tanh(W_g \cdot [h_{t-1}, x_t] + b_g)
    - **$i_t$ (Input Gate)**: 새로운 정보를 받아들일 비율 조절.
    - **$g_t$ (Cell Candidate)**: 새로운 정보 후보군.
3. **Cell State Update ($c_t$)**
    
    ct=ft⋅ct−1+it⋅gtc_t = f_t \cdot c_{t-1} + i_t \cdot g_t
    - **이전 기억($c_{t-1}$)을 얼마나 유지할지(f_t)** 결정.
    - **새로운 정보($g_t$)를 얼마나 반영할지($i_t$)** 결정.
4. **Output Gate ($o_t$)**
    
    ot=σ(Wo⋅[ht−1,xt]+bo)o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) ht=ot⋅tanh⁡(ct)h_t = o_t \cdot \tanh(c_t)
    - 최종적으로 출력할 hidden state 결정.

### **LSTM Cell 구조 요약**

각 time step에서 수행되는 연산을 정리하면:

|게이트|역할|
|---|---|
|**Forget Gate ($f_t$)**|이전 기억을 얼마나 유지할지 결정|
|**Input Gate ($i_t$)**|새로운 정보를 얼마나 반영할지 결정|
|**Candidate Cell ($g_t$)**|새로운 정보를 생성|
|**Cell State ($c_t$)**|장기 기억 저장소|
|**Output Gate ($o_t$)**|최종 출력될 hidden state 결정|

---

## **3. GRU (Gated Recurrent Unit)**

- LSTM의 변형된 버전으로, **더 가벼운 구조**.
- **Cell State($c_t$)와 Hidden State($h_t$)를 하나의 벡터로 통합**하여 계산량 감소.
- **Forget Gate와 Input Gate를 합쳐서 Update Gate ($z_t$)로 단순화**.

### **GRU 수식**

1. **Update Gate ($z_t$) - 이전 기억을 유지할지 결정** zt=σ(Wz⋅[ht−1,xt]+bz)z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
2. **Reset Gate ($r_t$) - 과거 정보를 얼마나 반영할지 결정** rt=σ(Wr⋅[ht−1,xt]+br)r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
3. **새로운 Hidden State 후보** h~t=tanh⁡(Wh⋅[rt⋅ht−1,xt]+bh)\tilde{h}_t = \tanh(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h)
4. **최종 Hidden State 계산** ht=(1−zt)⋅ht−1+zt⋅h~th_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t

### **LSTM vs GRU 비교**

|모델|특징|장점|단점|
|---|---|---|---|
|**LSTM**|Cell State + Hidden State|장기 기억 유지 가능|계산량 많음|
|**GRU**|Cell State 제거, Hidden State만 사용|계산량 적음, 빠름|장기 기억 유지 성능 LSTM보다 약간 떨어짐|

---

## **4. LSTM 정리**

|개념|설명|
|---|---|
|**기본 원리**|RNN의 단기 기억 문제를 해결하기 위해 Cell State 추가|
|**Forget Gate**|이전 기억을 얼마나 유지할지 결정|
|**Input Gate**|새로운 정보를 얼마나 반영할지 결정|
|**Output Gate**|최종 Hidden State를 결정|
|**Highway Network와 연결**|Residual Connection과 비슷한 방식으로 정보 유지|
|**GRU vs LSTM**|GRU는 LSTM을 경량화한 버전|

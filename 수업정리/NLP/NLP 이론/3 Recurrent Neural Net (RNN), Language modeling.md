
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



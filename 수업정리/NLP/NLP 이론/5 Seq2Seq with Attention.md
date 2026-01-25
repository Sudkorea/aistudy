## Seq2Seq 모델 소개 및 한계점
일련의 단어를 Encoder로 입력받아서, 변환된 단어를 Decoder로 출력함.
Encoder가 고정 크기의 Latent Vector로 변환하고, 이를 Decoder의 첫 State로 사용함

여기서 문제 : Bottleneck problem
고정된 크기의 Latent Vector는 긴 입력 정보를 모두 담는게 어려움



## Attention 개념, 적용법

Attention 점수 : 두 입력은 얼마나 관계있는가?

- Attention 분포를 활용하여 Weighted Sum 계산
- Attention 출력은 Attention 점수가 컸던 Hidden State 내용을 많이 담음

각 decoder Time Step마다 예측에 가장 도움주는 Encoder 내의 Hidden State에 집중하는 방법을 학습함

### Attention의 장점

기계번역 성능이 크게 향상
Bottleneck problem 해결
Vanishing Gradient 문제 발생하지 않음.

Attention은 사람이 해석 가능함. 

Attention 시각화

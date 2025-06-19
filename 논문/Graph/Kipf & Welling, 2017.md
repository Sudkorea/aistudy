# Abstract
graph-structured data semi-지도학습에 사용 가능한 scalable apporach를 소개함. 이는 그래프 구조에 맞게 변형한 CNN임. spectral graph convolution에서 국지적으로 1차근사하는거 영감받고 그 방식 써서 만듦. 우리 모델은 그래프의 vertex+edge 사이즈에 선형적이고, node의 feature, 국지적 그래프 구조를 반영하는 히든 레이어를 학습함. 실험 보면 이거 꽤 좋음.

## Semi-supervised learning?
GCN은 그래프의 각 노드에 주어진 feature가 일정하지 않음. Facebook에서 누구는 모든 정보를 기입하고, 누구는 아닌 것처럼. 따라서, 라벨이 전부 주어지지 않았지만, 또 누구는 주어졌으므로 semi-지도학습이라 부르는거임

# Conclusion
abstract에서 했던 말이랑 비슷함. 그래프 반지도학습에서 새로운 접근법을 알려드렸음. 우리 모델은 효율적인 layer-wise 전파 규칙을 제안하는데, 이게 sprectral convolutions on graph의 1차 근사를 base로 한 방법임. 실험들 보면 잘나와용~실험에 대한 이야기들(이건 나중가면 짜피 알게되니까 번역안함)

# Introduction
노드를 
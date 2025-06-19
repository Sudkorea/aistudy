# Abstract
graph-structured data semi-지도학습에 사용 가능한 scalable apporach를 소개함. 이는 그래프 구조에 맞게 변형한 CNN임. spectral graph convolution에서 국지적으로 1차근사하는거 영감받고 그 방식 써서 만듦. 우리 모델은 그래프의 vertex+edge 사이즈에 선형적이고, node의 feature, 국지적 그래프 구조를 반영하는 히든 레이어를 학습함. 실험 보면 이거 꽤 좋음.



# Introduction

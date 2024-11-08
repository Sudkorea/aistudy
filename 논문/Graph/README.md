공부를 이렇게 중구난방으로 하면, 누가 과연 이걸 흥미로운 포트폴리오라 생각하고 읽을것인가.
라고 생각하여 한 주제를 잡고 디깅하고자 함.

그래프에 대한 흥미를 바탕으로, 그래프 딥러닝 모델의 변천사를 쭉 공부해보자.

- **GC-MC (Graph Convolutional Matrix Completion)**
    
    - 논문: "Graph Convolutional Matrix Completion" (2017) - Berg et al.
    - GC-MC는 GCN을 활용하여 유저-아이템 행렬 완성을 시도한 초기 연구 중 하나입니다. GCN을 통해 유저와 아이템 간 상호작용을 학습하면서 잠재적인 평점 예측을 수행했습니다. 이는 NGCF 이전의 GCN을 추천 시스템에 적용하는 시초로 볼 수 있습니다.
- **PinSage**
    
    - 논문: "PinSage: A New Graph Convolutional Neural Network for Web-Scale Recommender Systems" (2018) - Ying et al.
    - PinSage는 대규모 그래프를 효율적으로 처리하기 위해 GraphSAGE와 비슷한 기법을 사용하여 샘플링을 활용했습니다. 특히 소셜 네트워크나 전자상거래 추천 시스템에 적합한 모델로, 큰 규모의 데이터에서도 빠르게 학습이 가능합니다.
- **GraphRec**
    
    - 논문: "GraphRec: Graph Neural Network for Social Recommendation" (2019) - Fan et al.
    - GraphRec은 추천 시스템에서 GNN의 가능성을 확장하여 사회적 관계를 이용한 추천 모델을 제시했습니다. 유저와 유저 간 관계까지 그래프 구조에 포함하여 추천 성능을 높이려 한 점이 특징입니다.
- **GCMC**
    
    - 논문: "Neural Collaborative Filtering" (NCF)와 함께 참고하면 좋습니다.
    - GCMC는 GC-MC와 달리 일반적인 행렬 완성보다는 상호작용 정보(implicit feedback)를 활용하여 좀 더 신경망 기반 접근 방식을 취합니다.
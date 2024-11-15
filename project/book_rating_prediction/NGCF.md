### Neural Graph Collaborative Filtering (NGCF) 논문 요약

NGCF는 그래프 신경망(GCN)을 추천 시스템에 최적화하기 위해 설계된 모델로, 사용자와 아이템 간의 상호작용을 깊이 반영하는 것을 목표로 함. 기존의 GCN 모델은 그래프의 구조적 정보는 잘 반영하지만, 노드 간의 상호작용(interaction)을 구체적으로 반영하지 않는 한계가 있음. NGCF는 이러한 한계를 극복하여, 사용자-아이템 간의 상호작용 정보를 효과적으로 반영하는 모델임.

#### 1. 상호작용 정보의 중요성
- 추천 시스템에서는 단순히 이웃 노드로부터 정보를 받아오는 것뿐만 아니라, 사용자와 아이템 간의 상호작용이 매우 중요함.
- NGCF는 각 노드가 자신의 이웃으로부터 단순히 정보를 받아오는 것에 그치지 않고, 노드 간 상호작용을 임베딩 연산에 반영함으로써 더 정교한 추천을 할 수 있음.

#### 2. 임베딩 프로파게이션
- NGCF는 그래프의 각 레이어에서 사용자와 아이템의 임베딩을 업데이트하며, 업데이트 과정에서 상호작용 정보가 포함됨.
- 각 레이어에서의 업데이트 수식은 다음과 같음:
  
  $e_u^{(l+1)} = \sigma \left( \sum_{v \in \mathcal{N}(u)} \left( e_u^{(l)} \odot e_v^{(l)} \right) W^{(l)} \right)$
  
  여기서 $e_u^{(l)}$는 $u$ 노드의 $l$-번째 레이어에서의 임베딩을, $W^{(l)}$는 학습 가능한 가중치 행렬을 의미함. $\odot$는 element-wise 곱셈을 나타내며, 이웃 노드와의 상호작용을 직접 반영함.

#### 3. 누적 임베딩
- NGCF는 각 레이어에서 학습된 임베딩을 누적하여 최종 임베딩을 생성함. 여러 레이어의 임베딩을 누적하면, 더 넓은 범위의 이웃 정보가 반영된 종합적인 임베딩을 얻을 수 있음.
- 최종적으로, 여러 레이어의 임베딩을 결합한 임베딩 벡터는 사용자와 아이템의 복합적인 특성을 잘 반영하여 추천 성능을 높임.

#### 4. 실험 결과
- 실험을 통해 NGCF는 기존의 협업 필터링 기법과 GCN 기반 모델보다 더 나은 성능을 보여줌.
- 특히, NGCF는 대규모 데이터셋에서도 효율적으로 작동하여 추천 시스템의 성능을 효과적으로 개선함을 확인함.

#### 요약
NGCF는 GCN의 구조를 추천 시스템에 맞게 최적화하여, 노드 간의 상호작용을 더욱 풍부하게 반영함으로써 추천 성능을 높인 모델임.

#### $\odot$ : element-wise 곱셈?
쉽게 말해서 행렬에서 곱셈을 각자 위치에서만 하는거
$$ \left[ \begin{matrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ \end{matrix} \right] \odot \left[ \begin{matrix} b_{11} & b_{12} \\ b_{21} & b_{22} \\ \end{matrix} \right] = \left[ \begin{matrix} a_{11}b_{11} & a_{12}b_{12} \\ a_{21}b_{21} & a_{22}b_{22} \\ \end{matrix} \right]$$
이걸 왜하는지 잘 생각해봤는데, 두 가지 이유가 있음.
1. 인접행렬 $(n, m)$ 위치에는 두 vertex n, m이 이루는 edge를 다룸. 행렬곱을 사용하면 이 위치에 $(n, 1), (n,2), ...$등 잡다한 곱이 많이 들어가는데, $\odot$을 사용하면 $(n, m)$ 위치에 대해서만 신경써줄 수 있기 때문(이게 맞는지는 몰겠음 뇌피셜)
2. 행렬곱은 $O(n^2)$이고 $\odot$는 $O(n)$라서 연산량이 확 낮아짐

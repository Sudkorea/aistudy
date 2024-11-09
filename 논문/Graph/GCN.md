## 0. Abstract

We present a **scalable approach** for semi-supervised learning on graph- structured data that is based on an efficient variant of **convolutional neural networks** which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized **first-order approximation** of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.



| The concept that I didn't know         | Explanation                                                                                                                                                                                                                                                                                                                                                                                  | 잘난척하는게 아니라 진짜 이해했냐?(What I actually get)                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Scalable Approach**                  | In machine learning, "scalable" means the model or method can handle large datasets efficiently. A scalable approach is designed to work well as the amount of data grows, without excessive increases in computation time or memory use.                                                                                                                                                    | scalable : 모델이 큰 데이터 셋을 효율적으로 다룰 수 있는.<br>approach니까, 데이터가 커져도 적당한 다항식 시간 안에 풀만한 |
| **CNN (Convolutional Neural Network)** | Although CNNs are commonly used in image processing, here, they are applied to graph data. CNNs process information by learning local patterns, and GCNs adapt this concept to graphs. Essentially, it means that GCNs "borrow" the structure of CNNs to process the relationships between nodes.                                                                                            | 합성곱신경망<br>시신경 모방한 그거<br><br>이건 따로 문서 파서 정리하는게 나을듯                                |
| **First-Order Approximation**          | This is a mathematical technique used to simplify calculations. In this paper, the authors use a first-order approximation to simplify how they aggregate information from a node’s neighbors, which makes the model faster and easier to compute, contributing to its scalability.                                                                                                          | 아..선형근사..                                                                        |
| Spectral Graph Convolutions            | a method for defining convolution operations on graphs. Since graphs don’t have a grid structure like images, standard convolution can’t directly apply. Instead, spectral graph convolutions operate in the **spectral domain** (frequency domain), where they use the graph's Laplacian matrix (a matrix that represents the structure of a graph) to capture relationships between nodes. | 그래프에서 합성곱 연산을 정의하는 방법.<br>라플라시안 구해서 푸리에 변환하고, 필터 적용한다함.                          |
| Citation Networks                      | a type of graph that represents the citation relationships among research papers.                                                                                                                                                                                                                                                                                                            | 논문 인용 네트워크. 초기 그래프 딥러닝 모델이 논문 생태계부터 시작됐다는게 신기함                                   |

## 1. Introduction
기존 graph-based semi-supervised learning에선 라플라시안 정규화가 사용되어, adjacency matrix와 neural network의 미분 가능한 함수를 결합한 손실 함수로 유클리드 거리를 최소화하려고 했음.

이 방식의 문제점 : 모델이 필요 없는 행을 너무 많이 가질 수 있음, 모델링 용량을 활용하지 못함.

따라서, 본 논문에서는 two-fold approach를 제안함.(두 가지 주요 아이디어 또는 절차를 결합한 접근 방식)

- Spectral Convolution과의 연결
  스펙트럴 도메인에서 그래프 구조를 이해하는 방법을 차용하나, 복잡한 계산은 선형근사로 때움. 즉, 모든 노드를 확인하지 않고, 이웃 노드 정보만 고려하여 연산량을 줄임
- GCN 도입
  각 노드가 자신이랑 이웃 노드들의 정보를 종합하여, 그래프 구조와 노드 속성을 반영할 수 있도록 함.

## 2. Fast Approximate Convolutions on Graphs

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

GCN 레이어의 출력, 즉 다음 레이어의 활성화 값(activations)을 나타냄

1. $H^{(l)}$: 현재 레이어 $l$의 노드 표현(activations). 각 노드의 특징 벡터가 들어 있는 행렬로, 이전 레이어의 출력을 의미함. $l=1$이면 이는 입력 특징 행렬 $X$를 나타냄
   
2. $W^{(l)}$ : 현재 레이어 $l$의 학습 가능한 가중치 행렬로, 노드 특징을 학습하면서 조정되는 파라미터

3. $\tilde{A} = A + I$ : 자기 자신 연결(Self-loop)을 추가한 인접 행렬. 그래프에 자기 자신과의 연결을 추가하여 각 노드가 자신에게도 정보를 전달할 수 있게 만들어줌.
   ($I$는 그럼 Identity matrix겠네)

5. **\( $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ \)**: 이 부분은 그래프 구조를 반영하기 위해 정규화된 인접 행렬을 계산하는 과정임. 노드 간 정보가 전달될 때, 각 노드의 연결 수에 따라 정규화하여 적절한 비율로 정보를 반영함. ($D$는 노드에 연결된)

6. $\sigma$ : 활성화 함수

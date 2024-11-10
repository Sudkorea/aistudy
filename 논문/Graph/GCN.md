## 0. Abstract

We present a **scalable approach** for semi-supervised learning on graph- structured data that is based on an efficient variant of **convolutional neural networks** which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized **first-order approximation** of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.



| The concept that I didn't know         | Explanation                                                                                                                                                                                                                                                                                                                                                                                  | 잘난척하는게 아니라 진짜 이해했냐?(What I actually get)                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Scalable Approach**                  | In machine learning, "scalable" means the model or method can handle large datasets efficiently. A scalable approach is designed to work well as the amount of data grows, without excessive increases in computation time or memory use.                                                                                                                                                    | scalable : 모델이 큰 데이터 셋을 효율적으로 다룰 수 있는.<br>approach니까, 데이터가 커져도 적당한 다항식 시간 안에 풀만한 |
| **CNN (Convolutional Neural Network)** | Although CNNs are commonly used in image processing, here, they are applied to graph data. CNNs process information by learning local patterns, and GCNs adapt this concept to graphs. Essentially, it means that GCNs "borrow" the structure of CNNs to process the relationships between nodes.                                                                                            | 합성곱신경망<br>시신경 모방한 그거<br><br>이건 따로 문서 파서 정리하는게 나을듯                                |
| **First-Order Approximation**          | This is a mathematical technique used to simplify calculations. In this paper, the authors use a first-order approximation to simplify how they aggregate information from a node’s neighbors, which makes the model faster and easier to compute, contributing to its scalability.                                                                                                          | 아..선형근사..                                                                        |
| Spectral Graph Convolutions            | a method for defining convolution operations on graphs. Since graphs don’t have a grid structure like images, standard convolution can’t directly apply. Instead, spectral graph convolutions operate in the **spectral domain** (frequency domain), where they use the graph's Laplacian matrix (a matrix that represents the structure of a graph) to capture relationships between nodes. | 그래프에서 합성곱 연산을 정의하는 방법.<br>라플라시안 구해서 푸리에 변환하고, 필터 적용한다함.                          |
| Citation Networks                      | a type of graph that represents the citation relationships among research papers.                                                                                                                                                                                                                                                                                                            | 논문 인용 네트워크. 초기 그래프 딥러닝 모델이 논문 생태계부터 시작됐다는게 신기함                                   |

# 1. Introduction
기존 graph-based semi-supervised learning에선 라플라시안 정규화가 사용되어, adjacency matrix와 neural network의 미분 가능한 함수를 결합한 손실 함수로 유클리드 거리를 최소화하려고 했음.

이 방식의 문제점 : 모델이 필요 없는 행을 너무 많이 가질 수 있음, 모델링 용량을 활용하지 못함.

따라서, 본 논문에서는 two-fold approach를 제안함.(두 가지 주요 아이디어 또는 절차를 결합한 접근 방식)

- Spectral Convolution과의 연결
  스펙트럴 도메인에서 그래프 구조를 이해하는 방법을 차용하나, 복잡한 계산은 선형근사로 때움. 즉, 모든 노드를 확인하지 않고, 이웃 노드 정보만 고려하여 연산량을 줄임
- GCN 도입
  각 노드가 자신이랑 이웃 노드들의 정보를 종합하여, 그래프 구조와 노드 속성을 반영할 수 있도록 함.

# 2. Fast Approximate Convolutions on Graphs

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

GCN 레이어의 출력, 즉 다음 레이어의 활성화 값(activations)을 나타냄

1. $H^{(l)}$: 현재 레이어 $l$의 노드 표현(activations). 각 노드의 특징 벡터가 들어 있는 행렬로, 이전 레이어의 출력을 의미함. $l=1$이면 이는 입력 특징 행렬 $X$를 나타냄
   
2. $W^{(l)}$ : 현재 레이어 $l$의 학습 가능한 가중치 행렬로, 노드 특징을 학습하면서 조정되는 파라미터

3. $\tilde{A} = A + I$ : 자기 자신 연결(Self-loop)을 추가한 인접 행렬. 그래프에 자기 자신과의 연결을 추가하여 각 노드가 자신에게도 정보를 전달할 수 있게 만들어줌.
   ($I$는 그럼 Identity matrix겠네)

5. $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ : 그래프 구조를 반영하기 위해 정규화된 인접 행렬을 계산하는 과정임. 노드 간 정보가 전달될 때, 각 노드의 연결 수에 따라 정규화하여 적절한 비율로 정보를 반영함. 

6. $\sigma$ : 활성화 함수

###
### 1. 그래프 라플라시안 $L$
스펙트럴 컨볼루션을 정의하기 위해 그래프 라플라시안(Laplacian) 행렬을 사용함. 그래프 $G = (V, E)$가 주어지면, 라플라시안 행렬 $L$은 다음과 같이 정의됨

$$L = D - A$$

### 2. 정규화 라플라시안 $\tilde{L}$ 
특정 노드에 편향되지 않도록 정규화하는 과정을 거친 것임.

$$\tilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$

잘 생각해보면 쉽게 유도 가능한데,

$L = D - A$
$\Leftrightarrow D^{-\frac{1}{2}}LD^{-\frac{1}{2}} = D^{-\frac{1}{2}}(D-A)D^{-\frac{1}{2}}$
$\Leftrightarrow \tilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$
$\therefore \tilde{L} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$

생각보다 별거 아니다.

### 3. 그래프 푸리에 변환과 고유벡터 분해
그래프 데이터를 주파수 영역으로 변환하기 위해, 라플라시안 $\tilde{L}$을 고유값 분해함

$$\tilde{L} = U \Lambda U^T$$

여기서:
- $U$: $\tilde{L}$의 eigenvector들로 구성된 행렬
- $\Lambda$: $\tilde{L}$의 eigenvalue들을 대각선으로 가지는 대각 행렬

이 고유벡터와 고유값을 이용해, 그래프의 푸리에 변환을 정의할 수 있음. 특정 그래프 신호 $x$의 푸리에 변환은 $U^T x$로 표현됨

### 4. 스펙트럴 필터 $g_\theta(\Lambda)$
스펙트럴 도메인에서의 필터링은 주파수 영역에서 신호를 필터링하는 것과 유사함. 필터 $g_\theta$는 고유값 행렬 $\Lambda$에 대해 다음과 같이 정의됨

$$g_\theta(\Lambda) = \sum_{k=0}^{K} \theta_k \Lambda^k$$

여기서:
- $\theta$: 학습할 필터 파라미터
- $K$: 근사의 차수

이 필터는 고유값 행렬 $\Lambda$에 대한 다항식을 사용해 특정 주파수 영역에서 정보를 추출하도록 설정됨

### 5. 스펙트럴 컨볼루션 연산 $g_\theta(L) * x$
스펙트럴 컨볼루션은 그래프의 신호 $x$에 필터 $g_\theta$를 적용하는 연산으로 정의됨

$$g_\theta(L) * x = U g_\theta(\Lambda) U^T x$$

- 그래프 신호 $x$를 푸리에 변환(주파수 영역으로 변환)하여, 고유값에 따라 정보를 필터링한 후 다시 원래의 도메인으로 변환함.
- $U$와 $U^T$는 푸리에 변환과 역변환을 수행하여, 필터링된 결과를 다시 그래프 구조에 맞게 적용함.

### 6. 1차 근사 (Chebyshev 다항식)
이 수식은 계산이 복잡하기 때문에, 선형근사를 사용해 간단히 만듦. 특히 Chebyshev 다항식을 사용하여 $g_\theta(L)$를 근사화할 수 있으며, $K=1$로 설정하여 다음과 같이 표현됨:

$$g_\theta(L) * x \approx \theta_0 x + \theta_1 \tilde{A} x$$

여기서 $\tilde{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$를 사용해 효율적인 계산이 가능해진다고함
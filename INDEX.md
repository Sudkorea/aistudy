# AISTUDY INDEX

> OpenClaw Agent용 요약 문서. 이 저장소의 학습 트랙, 논문 정리, 프로젝트 회고를 빠르게 검색할 수 있다.

**저장소 유형:** Public (AI/ML/Data Science 학습 기록)
**최종 갱신:** 2026-02-12

---

## REPOSITORY OVERVIEW

AI/ML/Data Science 학습 전과정을 기록한 저장소.
- ML/NLP/RecSys 수업 노트 (기초→심화)
- 논문 정리 (GCN, CF, PageRank, TDA, SASRec 등)
- 프로젝트 회고 (영화추천, 도서평점, 비트코인, 부동산, RAG)
- 모델 경량화/양자화 실험 기록
- MLOps/서빙 문서

---

## STUDY TRACKS

### A. Machine Learning 기초 (notes/수업정리/ML/)

14개 파일로 구성된 체계적 ML 커리큘럼.

| 파일 | 핵심 내용 | 키워드 |
|------|-----------|--------|
| `1. ML LifeCycle.md` | Planning→Data Prep→Model→Eval→Deploy→Monitor | #ML생애주기 #Transfer_Learning |
| `2. Linear Regression.md` | OLS, MSE/RMSE/MAE/R², k-NN | #회귀 #평가지표 |
| `3. Loss Function.md` | Cross-entropy, MSE, 정규화 | #손실함수 #최적화 |
| `4. Classifier.md` | 로지스틱 회귀, 결정 경계 | #분류 |
| `5. 가중치.md` | 가중치 초기화 기법 | #초기화 |
| `6. Linear Model.md` | 특성 공학, 정규화 | #선형모델 |
| `7. Neural Networks.md` | Perceptron, MLP, 활성화함수(Sigmoid/Tanh/ReLU/ELU) | #신경망 #MLP |
| `8. Backpropagation.md` | 체인룰, 기울기 소실/폭발, Adam/Momentum/Nesterov | #역전파 #최적화기 |
| `9. Activation Functions.md` | 활성화 함수 상세 비교 | #활성화함수 |
| `10. Weight Initialization.md` | Xavier, He 초기화 | #초기화 |
| `11. Learning Rate Scheduling.md` | LR decay 전략 | #학습률 |
| `12. Data Preprocessing.md` | 정규화, 표준화, 결측치 | #전처리 |
| `13. Data Augmentation.md` | 데이터 증강 기법 | #증강 |
| `14. Transformer.md` | Self-Attention, Multi-Head, Positional Encoding, Q/K/V | #Transformer #Attention |

### B. NLP (notes/수업정리/NLP/)

**NLP 이론/ (8개):**

| 파일 | 핵심 내용 |
|------|-----------|
| `1 Tokenization.md` | BPE, WordPiece, 문자 수준 토큰화 |
| `2 Word Embedding.md` | Word2Vec, GloVe, 문맥 임베딩 |
| `3 Recurrent Neural Net.md` | RNN, 언어 모델링, 기울기 소실 |
| `4 Long Short Term Memory.md` | LSTM/GRU 게이트 메커니즘 |
| `5 Seq2Seq with Attention.md` | 인코더-디코더, Teacher Forcing, Attention |
| `6 Transformer.md` | Scaled dot-product attention, masking, layer norm |
| `7 Self-Supervised Pre-trained Model, BERT.md` | MLM, 양방향 인코딩, 다운스트림 태스크 |
| `8 Decoder.md` | GPT 스타일 자기회귀 생성 |

**MRC/ (4개):** MRC 개요, Extraction-based, Generation-based, Sparse Embedding(TF-IDF, BM25)

**NLP 기초 프로젝트/ (4개):** NLP 처리 문제, 텍스트 데이터 분석, Huggingface, 모델 성능 분석

### C. 추천시스템 (notes/수업정리/RecSys/)

**RecSys 기초 프로젝트/:**

| 파일 | 핵심 내용 |
|------|-----------|
| `2 Collaborate Filtering.md` | User/Item-based CF, K-NN, 유사도(MSD/Cosine/Pearson/Jaccard), 평점 예측 |
| `3 Item2Vec, ANN.md` | 아이템 임베딩, 근사 최근접 이웃 |

**ML for RecSys/ (8개):**

| 파일 | 핵심 내용 |
|------|-----------|
| `1. Recommendation System.md` | CF vs Content-based 개요 |
| `2. 통계학 기본.md` | 확률분포, 통계적 추론 |
| `3. Generative Model.md` | VAE, GAN for RecSys |
| `4. VI.md` | 변분 추론 |
| `5. Monte Carlo Approx..md` | MCMC, 베이지안 샘플링 |
| `6. Data Attribution.md` | 모델 예측 해석 |
| `7 인과성과 기계학습.md` | 처리 효과, 반사실적 추론 |
| `Markov Chain.md` | 전이 확률, 순차 모델링 |

### D. PyTorch (notes/수업정리/pytorch/)

6개 파일: week1 기초, 일일 실습(240805~240808), C++ LibTorch 텐서 연산

### E. 모델 최적화/경량화 (notes/수업정리/최적화, 경량화/)

| 파일 | 핵심 내용 | 키워드 |
|------|-----------|--------|
| `1. introduction.md` | 모델 효율화 개요 | #경량화 |
| `2. Pruning.md` | 구조적/비구조적 프루닝 | #Pruning |
| `3. Knowledge Distillation.md` | Teacher-Student 전이 | #증류 |
| `4. Quantization.md` | FP32→INT8/INT4, 선형 양자화 매핑 | #양자화 |
| `5. PEFT.md` | LoRA, Adapter, Q-LoRA | #PEFT #LoRA |
| `6. Distributed Training.md` | 데이터/모델 병렬화 | #분산학습 |

---

## PAPERS (notes/논문/)

### Graph Neural Networks (Graph/)

| 논문 | 핵심 | 키워드 |
|------|------|--------|
| `NGCF.md` | GCN으로 user-item 그래프 모델링, element-wise product, 다층 임베딩 전파 | #GCN #RecSys #그래프 |
| `GCN.md`, `Kipf & Welling, 2017.md` | 그래프 컨볼루션 기초 | #GCN |
| `Graph.md` | 그래프 이론 기초 (노드, 엣지, 인접행렬) | #그래프이론 |

### Collaborative Filtering (Collaborative Filtering/)

| 논문 | 핵심 | 키워드 |
|------|------|--------|
| `Collaborative Filtering for Implicit Feedback Datasets.md` | 신뢰도 행렬 C=1+αR, ALS 최적화, 희소 데이터 처리, 선호행렬 P(이진) vs 관측행렬 R | #ALS #암묵피드백 #행렬분해 |
| `개요.md` | CF 유형 개요 | #CF |

### 기타 논문

| 논문/폴더 | 핵심 | 키워드 |
|-----------|------|--------|
| `PageRank/PageRank.md` | 그래프 랭킹, 감쇠계수 d=0.85, Random Surfer, Personalized PageRank | #PageRank #고유벡터 |
| `TDA/비전공자도...TDA개념 차근차근정리.md` | Persistent Homology, Betti수(β₀:연결, β₁:루프, β₂:공동), 바코드/퍼시스턴스 다이어그램, 응용(단백질/EEG/금융/RecSys) | #TDA #위상수학 #Betti |
| `self_attention_in_recsys/SASRec.md` | 순차 추천에 Self-Attention, MC/RNN 절충, 인과 마스킹, O(n²d) 복잡도 | #SASRec #순차추천 |
| `LightGBM/LightGBM.md` | XGBoost(균형)↔LightGBM(불균형), 부스팅은 고잔차 샘플 집중 | #LightGBM #앙상블 |
| `bitnet/` | Binary neural networks (PDF) | #BitNet |

---

## PROJECTS (projects/)

### Movie Recommendation (13~15주차)
- 데이터팀→모델팀 전환, 모델-데이터 상호작용 학습

### Book Rating Prediction
- **핵심 인사이트:** 순차 모델링 ≠ 시계열 (x_i→x_{i+1} 관계), 임베딩 ≈ 위상학적 immersion (차원축소+국소구조 보존)

### Bitcoin Price Prediction (7주차)
- 시계열 예측, 미래 정보 마스킹 (causal attention과 유사)

### House Rent Prediction
- FT-Transformer로 정형 데이터 회귀

### RAG Experiments
- Retrieval-Augmented Generation 탐색

### 금융상품 추천 (NCP/우체국)
- 레거시 ALS/K-means/BPR 분석, 고정 하이퍼파라미터 문제 발견, MLOps/자동 튜닝 제안

---

## PROGRESS & PRESENTATIONS (posts/progress/)

| 파일 | 내용 | 키워드 |
|------|------|--------|
| `MAMBA.md` | SSM, Transformer 한계(O(n²)), Impossible Triangle, HiPPO→LSSL→S4→Mamba, 선택적 상태 갱신 | #Mamba #SSM |
| `MAMBA-발표자료.md` | Mamba 발표 구조 | #발표 |
| `SALMONN-모델분석.md` | 음성-오디오 멀티모달 모델 분석 (600+줄) | #SALMONN #멀티모달 |
| `recsys als 발표내용.md` | ALS 레거시 분석, K-means+ALS 업셀링, BPR 실험 제안 | #ALS #발표 |
| `5. Recommendations with Side-information.md` | 메타데이터 활용 추천 | #사이드정보 |
| `역전파.md` | 역전파 설명 | #역전파 |
| `팀의 문제 해결 경험(러닝 커브).md` | 팀 문제해결 회고 | #협업 #회고 |
| `일일학습로그-20241209.md` | 일일 학습 기록 | #일일 |

---

## TECHNICAL DOCS (docs/)

### 양자화 실험 (quantization/실험일지/)
- `250122.md`: ASR 모델 전략적 양자화 - 우선순위: encoder self-attention, decoder FFN, 중간층 / 회피: 최종출력층, 첫 입력층 / 목표: 2x 메모리 감소+지연시간 개선
- `250123.md`, `250124.md`, `250124_split.md`: 후속 실험

### Product Serving
- `요약.md`: Batch vs Online 서빙, 디자인 패턴(Batch/Web Single/Sync/Async), REST API
- `ubuntu docker 실행법.md`: Docker 배포 가이드

### 기타
- `POD miniconda 구성 가이드라인.md`: 환경 설정
- `troubleshooting/`: CUDA Byte 에러, pandas CSV 인코딩

---

## CONCEPTS & MISC

| 파일 | 내용 |
|------|------|
| `notes/개념정리/브로드캐스팅.md` | NumPy 브로드캐스팅 규칙 |
| `notes/dataviz/pandas-matplotlib-정리.md` | pandas/matplotlib 시각화 |
| `notes/dataviz/Albumentations.md` | 이미지 증강 라이브러리 |
| `notes/책/통계학.md` | 통계학 개념 |
| `notes/책/ADSP 준비.md` | 데이터분석 자격증 |
| `notes/책/정처기.md` | 정보처리기사 |
| `notes/책/양자컴/양자컴퓨터1.md` | 양자컴퓨팅 기초 |
| `notes/TDA Case Study.md` | TDA 실제 사례 |

---

## CROSS-REFERENCES

### Transformer 계열
- 기초: `ML/14. Transformer.md` ↔ `NLP 이론/6 Transformer.md`
- 응용: SASRec 논문, BERT 노트
- 대안: `posts/progress/MAMBA.md` (SSM)

### 추천시스템 전체 흐름
- 고전: `RecSys 기초 프로젝트/2 Collaborate Filtering.md`
- 신경망: NGCF 논문, SASRec 논문
- 실무: Movie/Book 프로젝트, 금융상품 추천(NCP)

### 최적화→배포
- 이론: `최적화, 경량화/` 6개 파일
- 실험: `docs/quantization/실험일지/`
- 서빙: `docs/Product Serving/`

### 그래프 방법론
- 이론: PageRank, GCN
- RecSys 적용: NGCF
- 위상학: TDA

### 순차 모델링 진화
- RNN→LSTM/GRU→Transformer→Mamba(SSM)

---

## LEARNING PROGRESSION

1. **ML 기초:** 선형모델, 신경망, 최적화, 손실함수
2. **딥러닝 심화:** 역전파, 활성화함수, 초기화, 정규화
3. **NLP 특화:** RNN→LSTM→Transformer→BERT
4. **RecSys 특화:** CF→행렬분해(ALS/BPR)→신경망/그래프
5. **고급 주제:** 경량화(양자화/프루닝), TDA, Mamba/SSM, 프로덕션 배포

---

## QUICK SEARCH GUIDE

| 질문 | 참조 파일 |
|------|-----------|
| ALS 작동 원리 | `notes/논문/Collaborative Filtering/Collaborative Filtering for Implicit Feedback Datasets.md` |
| Self-Attention 설명 | `notes/수업정리/NLP/NLP 이론/6 Transformer.md` |
| NGCF란? | `notes/논문/Graph/NGCF.md` |
| 과거 RecSys 프로젝트 | `projects/project/Movie_recommendation/`, `posts/progress/recsys als 발표내용.md` |
| 양자화 전략 | `docs/quantization/실험일지/250122.md` |
| Mamba/SSM 이해 | `posts/progress/MAMBA.md` |
| TDA 개념 | `notes/논문/TDA/비전공자도...TDA개념 차근차근정리.md` |
| 서빙 패턴 | `docs/Product Serving/요약.md` |
| LoRA/PEFT | `notes/수업정리/최적화, 경량화/5. PEFT.md` |
| 순차 추천(SASRec) | `notes/논문/self_attention_in_recsys/Self-Attentive Sequential Recommendation.md` |

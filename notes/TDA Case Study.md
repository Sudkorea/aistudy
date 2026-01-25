# 1) 문제정의 & 가설

- **비즈니스 목표**: Recall@5 개선이 목적. 차후 협의해야함
    
- **TDA 가설**: 어떤 위상 신호가 유용할까?
    
    - 행동공간의 **manifold cluster** (유사 유저 군집)
        
    - **loops**(주기적 패턴), **holes**(미탐색 상품군의 공백)
        
    - **regime change**(시계열 상태 전이) 포착
        

# 2) 데이터 스키마 & 스케일

- **엔티티/관계**: Users–Items(예금/적금/보험)–



    
- **볼륨/희소성**: 노드·엣지 수, 로그 기간, 결측/편향
    
- **민감정보**: 데이터 암호화로 유저, 아이템 ID 막아둠

# 3) 표현(Representation) 선택

- **Point cloud**: 고객 프로필(수치/범주 임베딩) → distance space
    
- **Graph**: user–item bipartite / user–user similarity / item–item co-occurrence
    
- **Time series / sequences**: 세션/일자별 이벤트 시계열
    
- **선택 근거**: 고차원/희소성/시간축 존재 여부, 연산 복잡도
    

# 4) Complex & Filtration 설계

- **Vietoris–Rips / Alpha / Witness / Clique complex** 중 무엇을? (그래프면 보통 **Clique complex**, 임베딩 공간이면 **VR/Alpha**, 대규모면 **Witness**)
    
- **Filtration**: 거리/가중치 임계, kNN, 시간(**zigzag persistence**, **vineyards**), 금액/위험도 계층
    
- **파라미터**: max homology dim (보통 H₀, H₁부터), 샘플·landmark 수, kNN-k, 거리척도(예: Gower/learned metric)
    

# 5) TDA 기법 & 특성화

- **Invariants**: Betti numbers, connected components(H₀), cycles(H₁), (필요시 H₂)
    
- **Persistence diagram** → 벡터화: **persistence images**, **landscapes**, **silhouettes**
    
- **Diagram kernels**: sliced Wasserstein, persistence Fisher 등
    
- **Mapper**: lens(UMAP/Isomap/PCA/샤프니스/centricity), cover(resolution, gain) 선택
    

# 6) ML 통합 전략(우리 추천 모델 관점)

- **Feature augmentation**: TDA 임베딩을 **Two-Tower** side features, **LightGCN**/NGCF의 user/item feature로 주입
    
- **Distance/Kernel 교체**: diagram kernel로 유사도 정의(최근접 이웃·retrieval 단계)
    
- **Clustering**: Mapper/VR-H₀ 기반 세그먼트로 **리랭킹 규칙**(상품군 다양성·리스크 제한) 적용
    
- **Regularization**: topological loss(예: loop 유지/해소)로 표현 학습 제어
    
- **Ablation**: “Base → +TDA(그래프) → +TDA(시계열) → 둘 다” 단계별 비교
    

# 7) 시각화 산출물(의사결정용)

- **Mapper graph**: 고객군/전환율/민원률 색칠 → 인사이트 스토리라인
    
- **Persistence diagram** & **landscape**: 세그먼트·기간별 스택 비교
    
- **Graph explanations**: saliency(어떤 엣지/노드가 특정 loop에 기여?) 역추적 리포트
    

# 8) 검증 계획(오프라인/온라인)

- **Offline**: Recall@K, NDCG@K, Precision@K, Calibration, Item Coverage, Diversity/Gini, Popularity bias
    
- **Segment eval**: 연령·위험등급·상품군별 편향/효과
    
- **Online A/B**: 전환·가입금액·이탈률·CS콜 변화, **guardrails**(리스크/불완전판매)
    
- **통계적 안정성**: bootstrap으로 diagram distance(bottleneck/Wasserstein) 신뢰구간
    

# 9) 스케일 & 운영

- **라이브러리**: giotto-tda, GUDHI, Ripser.py, scikit-tda, KeplerMapper(+ PyTorch/Lightning 통합 가능성)
    
- **복잡도/리소스**: n=O(10⁵~10⁶)일 때 샘플링/landmark/witness, kNN-graph 전처리(FAISS/Annoy)
    
- **파이프라인화**: 재현 가능한 end-to-end(시드, 버전, 파라미터 기록), 배치/스케줄링
    

# 10) 리스크 & 품질

- **데이터 누수**(미래 로그 포함 금지), **과적합**(파라미터 스윕), **해석 과잉**(노이즈 vs 의미)
    
- **공정성/규제**: 세그먼트 차별, 설명가능성(Mapper로 스토리 제공)
    
- **보안/개인정보**: 匿名화, 최소수집, 접근통제
    

# 11) 케이스 비교-평가 루브릭 (점수화)

- **Problem fit(0-3)**: 우리 과제와 문제정의 유사도
    
- **Data fit(0-3)**: 데이터 구조/스케일 유사도
    
- **Evidence strength(0-3)**: A/B·대조군·오픈소스 재현성
    
- **Transfer cost(0-3)**: 도입 복잡도(인력/인프라/시간)
    
- **ROI potential(0-3)**: KPI 영향·리스크 감소 기대  
    → 총점으로 Top-N 케이스 선정
    

# 12) 도메인별 “무엇을 찾아볼지” 키워드

- **금융 추천/리스크**: “topological features for recommender systems”, “Mapper for customer segmentation”, “transaction graph TDA”, “zigzag persistence user behavior”
    
- **시계열**: “time-delay embedding + persistence”, “state transition detection with TDA”
    
- **그래프**: “clique complex recommender”, “VR complex on node2vec embeddings”
    
- **해석/커널**: “persistence landscape for classification”, “persistence Fisher kernel recommender”
    

# 13) 우리 데이터 유형별 권장 TDA 레시피

- **고객 프로필(tabular)**: metric(예: Gower 또는 learned metric) → **VR complex(H₀,H₁)** → **persistence images** → 세그먼트/특징
    
- **유저-아이템 그래프**: bipartite → item/item co-occurrence로 **weighted graph** → **Clique complex** 또는 **VR on embeddings** → H₀/H₁ → loop 기반 다양성 신호
    
- **행동 시계열**: 세션별 이벤트율 → **time-delay embedding** → **VR/Alpha** → regime change 탐지(zigzag) → 콜드스타트/리텐션 피처
    
- **대규모 스케일**: **Witness complex/landmark sampling**, kNN-graph(ANN), H₀ 중심 후 H₁ 점진 확장
    

# 14) 통계적 타당성 체크리스트

- **Stability**: subsample/perturbation에서 diagram distance 분포
    
- **Permutation test**: 레이블 셔플 후 성능 차 검정
    
- **Multiple hypothesis control**: 세그먼트 다수 비교 시 FDR 관리
    

# 15) 케이스 스터디 문서 포맷(1~2p/사례)

1. **요약**(문제/데이터/결과)
    
2. **데이터 & 표현**(graph/point/time, metric)
    
3. **Complex & filtration**(선택/파라미터)
    
4. **TDA 산출물**(diagram/landscape/Mapper)
    
5. **ML 통합 방법**(feature/kernel/regularizer/리랭킹)
    
6. **평가**(오프라인/온라인, 통계)
    
7. **인프라/비용**(연산시간/메모리/툴)
    
8. **리스크/한계**(해석/스케일/규제)
    
9. **이식 포인트**(바로 가져올 수 있는 파트) & **To-do**
    

# 16) 2주 PoC 로드맵(권장)

- **Week 1**:
    
    - 샘플(유저≤100k, 아이템≤5k) 추출 → kNN-graph/임베딩 생성
        
    - **H₀/H₁** 기반 TDA 피처 벡터화(giotto-tda)
        
    - **Base 모델 vs +TDA 피처** 오프라인 비교(Recall@20, NDCG@20, Coverage)
        
- **Week 2**:
    
    - **Mapper**로 세그먼트 도출 & 비즈니스 설명자료 제작
        
    - **Ablation & Stability**(bootstrap) → 채택 기준 정리
        
    - 온라인 실험 설계서(대상, 노출규칙, guardrails)
        

---

원하면 다음 단계로 **후보 케이스 목록(논문/산업 적용 기사/오픈소스 리포)**을 실제 검색해 정리해줄게. 그때는 위 포맷에 맞춰 5~8건을 채워서, 우리 프로젝트에 **즉시 이식 가능한 부분**만 추려 제공하겠다.
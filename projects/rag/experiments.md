# RAG 기반 법률 MCQ 실험 보고서

> 한국 법률 객관식 문제(259문항, dev.csv) 대상 RAG 파이프라인 최적화 실험 20종의 아이디어, 이론적 배경, 처리 과정, 핵심 코드를 정리한 문서.

## 실험 결과 요약

| Stage | 방법론 | 정확도 (dev) | 실행시간 | 비고 |
|-------|--------|:------------:|:--------:|------|
| 0 | Baseline RAG | 52.90% | 64.3s | Dense retrieval + Structured Output |
| 1 | OADR (Options-Aware Dense Retrieval) | 50.97% | 101.5s | 질문/전체 쿼리 이중 검색 |
| 2 | **ParSeR (Parametric Provision Prediction)** | **55.60%** | **168.6s** | **최고 단일 방법론** |
| 3 | Legal Reasoning (IRAC + Few-shot) | 54.83% | 204.1s | IRAC 프레임워크 |
| 4 | Log-Likelihood 기반 추론 | 54.83% | 193.7s | logprobs로 토큰 확률 비교 |
| 5 | Self-Verification | 54.44% | 257.3s | 2단계 추론 (생성→검증) |
| 6 | SAC + Hybrid Dense/BM25 | 52.90% | 188.3s | RRF 기반 하이브리드 검색 |
| 7 | PoE + Token Bias Debiasing | 48.65% | 179.3s | 편향 보정 실패 |
| 8 | Adaptive RAG + CISC | 47.49% | 379.4s | 신뢰도 기반 라우팅 |
| 9 | Chain-of-Thought (CoT) | 52.12% | 484.7s | 단계적 추론 |
| 10a | Ensemble Top-3 (2,3,4) | 54.05% | 0.0s | 오프라인 다수결 |
| 10b | Ensemble Top-5 (0,2,3,4,5) | 54.44% | 0.0s | 오프라인 다수결 |
| 10c | Ensemble Diverse (2,3,0) | 55.21% | 0.0s | 다양성 기반 다수결 |
| 11 | Focused Retrieval | 53.67% | 302.7s | 문서 수 축소 (7→4) |
| 12 | Dynamic Few-Shot | 50.58% | 350.3s | 검색 문서를 few-shot으로 변환 |
| 13 | PRIdE Debiasing | 37.84% | 3447.1s | 과도한 보정으로 실패 |
| 14 | Binary Task Decomposition | N/A | >600s | 시간 초과 |
| 15 | Adaptive Routing | N/A | >600s | 시간 초과 |
| 16 | Negative-Aware CoT Hybrid | N/A | >600s | 시간 초과 |
| 17 | Negative-Aware Prompt | 52.51% | 172.0s | 부정형 전용 프롬프트 |
| 18 | Anti-Position-Bias Prompt | 55.21% | 177.1s | 위치 편향 경고 |
| 19 | Combined (17+18) | 54.44% | 170.1s | 두 전략 결합 |
| 20a | Type-Weighted Voting | 54.44% | 0.0s | 유형별 가중 투표 |
| **20b** | **Best-per-Type Routing** | **57.92%** | **0.0s** | **전체 최고 (dev 과적합 주의)** |
| 20c | Diverse Ensemble | 55.60% | 0.0s | 다양성 기반 앙상블 |

### 테스트 세트 검증 (비중복 243문항)

| 방법론 | dev 정확도 | test 정확도 | 차이 |
|--------|:---------:|:----------:|:----:|
| 4_loglikelihood | 54.83% | **53.09%** | -1.74%p |
| 2_parser | 55.60% | 51.44% | -4.16%p |
| 5_self_verification | 54.44% | 51.03% | -3.41%p |
| 18_anti_bias | 55.21% | 50.62% | -4.59%p |
| 20b best-per-type | 57.92% | 53.50% | -4.42%p |

> **최종 제출**: Stage 4 (Log-Likelihood) - dev/test 간 가장 일관된 성능, 과적합 최소

---

## 목차

- [Stage 0: Baseline RAG](#stage-0-baseline-rag)
- [Stage 1: OADR](#stage-1-oadr-options-aware-dense-retrieval)
- [Stage 2: ParSeR](#stage-2-parser-parametric-provision-prediction)
- [Stage 3: Legal Reasoning Prompting](#stage-3-legal-reasoning-prompting)
- [Stage 4: Log-Likelihood 기반 추론](#stage-4-log-likelihood-기반-추론)
- [Stage 5: Self-Verification](#stage-5-self-verification)
- [Stage 6: SAC + Hybrid Dense/BM25](#stage-6-sac--hybrid-densebm25-search)
- [Stage 7: PoE + Token Bias Debiasing](#stage-7-product-of-experts-poe--token-bias-debiasing)
- [Stage 8: Adaptive RAG + CISC](#stage-8-adaptive-rag--cisc-confidence-weighted-self-consistency)
- [Stage 9: Chain-of-Thought](#stage-9-chain-of-thought-cot--parser-retrieval)
- [Stage 10: Offline Ensemble](#stage-10-offline-ensemble-majority-vote)
- [Stage 11: Focused Retrieval](#stage-11-focused-retrieval-fewer-better-docs)
- [Stage 12: Dynamic Few-Shot](#stage-12-dynamic-few-shot-learning)
- [Stage 13: PRIdE Debiasing](#stage-13-pride-debiasing--length-normalization)
- [Stage 14: Binary Task Decomposition](#stage-14-binary-task-decomposition)
- [Stage 15: Adaptive Routing](#stage-15-adaptive-routing-with-confidence-weighted-voting)
- [Stage 16: Negative-Aware CoT Hybrid](#stage-16-negative-aware-cot-hybrid-two-pass)
- [Stage 17: Negative-Aware Prompt](#stage-17-negative-aware-prompt-parser)
- [Stage 18: Anti-Position-Bias Prompt](#stage-18-anti-position-bias-prompt-parser)
- [Stage 19: Combined (17+18)](#stage-19-combined-negative-aware--anti-bias-parser)
- [Stage 20: Smart Offline Ensembles](#stage-20-smart-offline-ensembles)
- [종합 분석](#종합-분석)

---

## Stage 0: Baseline RAG

### 아이디어
질문과 유사한 기출문제를 벡터 검색으로 찾아 컨텍스트로 제공하고, LLM이 이를 참고하여 정답을 추론하는 기본 RAG 파이프라인.

### 이론적 배경
RAG(Retrieval-Augmented Generation)는 외부 지식을 검색하여 LLM의 생성 과정에 주입하는 기법이다. 순수 LLM은 학습 데이터에만 의존하지만, RAG는 벡터 데이터베이스에서 관련 문서를 검색(Retrieval)하여 컨텍스트로 제공함으로써 도메인 특화 지식을 보강(Augmentation)하고 환각(hallucination)을 감소시킨다. 임베딩 모델로 문서를 벡터화하고, 코사인 유사도 기반 최근접 이웃 탐색으로 관련 문서를 찾는다.

### 처리 과정
1. 사용자 질문을 OpenAI `text-embedding-3-small` 모델로 임베딩 벡터로 변환
2. ChromaDB에서 코사인 유사도 기반으로 상위 5개 유사 기출문제 검색
3. 검색된 문서를 번호 매긴 참고자료 블록으로 구성
4. 시스템 프롬프트 + 참고자료 + 질문을 `gpt-4o-mini`에 전달
5. Structured Output(Pydantic `AgentResponse`)으로 A/B/C/D 중 하나의 정답 파싱

### 핵심 코드

```python
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    """query와 유사한 문서를 ChromaDB에서 검색"""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    documents: list[str] = results["documents"][0] if results["documents"] else []
    return documents

def generate(query: str, context_docs: list[str]) -> AgentResponse:
    """검색된 컨텍스트와 질문을 LLM에 전달"""
    context_block = "\n\n---\n\n".join(
        f"[참고 {i + 1}]\n{doc}" for i, doc in enumerate(context_docs)
    )
    user_message = f"## 참고자료 (유사 기출문제)\n{context_block}\n\n## 풀어야 할 문제\n{query}"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=AgentResponse,
        temperature=0.0,
    )
    return response.choices[0].message.parsed

def answer_question(query: str) -> AgentResponse:
    """RAG 전체 파이프라인: 검색 → 생성"""
    docs = retrieve(query)
    return generate(query, docs)
```

### 결과
- **정확도**: 52.90%
- **실행시간**: 64.3초

---

## Stage 1: OADR (Options-Aware Dense Retrieval)

### 아이디어
질문 텍스트만으로 검색한 결과와 선택지를 포함한 전체 쿼리로 검색한 결과를 병합하여, 보다 다양한 관점의 컨텍스트를 확보한다.

### 이론적 배경
Multi-Query Retrieval은 동일한 정보 요구를 여러 방식으로 표현하여 검색 다양성을 높이는 기법이다. 법률 객관식 문제의 경우, 질문 자체는 쟁점을 나타내고 선택지는 구체적 판단 기준을 포함한다. 질문만으로 검색하면 개념적으로 유사한 문제를 찾지만, 선택지를 포함하면 특정 조항이나 판례가 언급된 문제를 찾을 수 있다. 두 검색 결과를 병합하면 검색 커버리지가 증가하고 관련 문서 누락 가능성이 감소한다.

### 처리 과정
1. 입력 쿼리에서 첫 번째 빈 줄 이전까지를 질문 부분으로 추출 (`_extract_question`)
2. 질문 텍스트만으로 ChromaDB에서 상위 5개 문서 검색
3. 전체 쿼리(질문 + 선택지)로 ChromaDB에서 상위 5개 문서 검색
4. 두 검색 결과를 중복 제거하며 병합 (전체 쿼리 결과 우선, 최대 7개)
5. 병합된 문서를 컨텍스트로 `generate` 호출하여 답변 생성

### 핵심 코드

```python
def _extract_question(query: str) -> str:
    """query에서 질문 부분만 추출 (첫 번째 빈 줄 이전)"""
    parts = query.split("\n\n", 1)
    return parts[0]

def answer_question_oadr(query: str) -> AgentResponse:
    """OADR: 질문 단독 검색 + 전체 쿼리 검색 병합"""
    question_only = _extract_question(query)

    # 두 가지 관점으로 검색
    docs_by_question = retrieve(question_only, top_k=5)
    docs_by_full = retrieve(query, top_k=5)

    # 중복 제거하며 병합 (순서 유지, 최대 7개)
    seen = set()
    merged: list[str] = []
    for doc in docs_by_full + docs_by_question:  # full query 결과 우선
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= 7:
            break

    return generate(query, merged)
```

### 결과
- **정확도**: 50.97%
- **실행시간**: 101.5초

---

## Stage 2: ParSeR (Parametric Provision Prediction)

### 아이디어
LLM이 먼저 문제를 풀기 위해 필요한 법률 조항을 예측하고, 예측된 조항으로 추가 검색을 수행하여 더 정밀한 법률 컨텍스트를 확보한다.

### 이론적 배경
Parametric Search and Retrieval은 LLM의 파라메트릭 지식(학습된 내부 지식)을 활용하여 검색 쿼리를 개선하는 기법이다. 법률 문제는 특정 법률 조항과 밀접히 연관되어 있으며, LLM은 사전 학습을 통해 법률 용어와 조항 간 연관성을 학습했다. 문제를 읽고 관련 조항을 예측하는 것은 Query Expansion의 일종으로, 원본 쿼리만으로는 찾기 어려운 조항 중심의 문서를 검색할 수 있게 한다. 이는 HyDE(Hypothetical Document Embeddings)와 유사하게, 관련 있을 것으로 예상되는 텍스트를 생성하여 검색에 활용하는 방식이다.

### 처리 과정
1. LLM(`gpt-4o-mini`)에게 문제를 제시하고 관련 법률명과 조항 번호를 예측하도록 요청 (`_predict_provisions`)
2. 질문 텍스트만으로 상위 3개 문서 검색
3. 전체 쿼리로 상위 3개 문서 검색
4. 예측된 조항 텍스트로 상위 3개 문서 검색
5. 세 검색 결과를 중복 제거하며 병합 (조항 검색 결과 최우선, 최대 7개)
6. 병합된 문서를 컨텍스트로 답변 생성

### 핵심 코드

```python
PROVISION_PROMPT = """\
당신은 한국 법률 전문가입니다.
아래 법률 객관식 문제를 읽고, 이 문제를 풀기 위해 참조해야 할 법률명과 관련 조항을 예측하세요.
규칙:
- 법률명과 조항 번호를 구체적으로 나열하세요.
- 예: "형사소송법 제232조", "헌법 제111조"
- 최대 3개까지만 나열하세요."""

def _predict_provisions(query: str) -> str:
    """LLM에게 관련 법률 조항을 예측하게 한다."""
    client = OpenAI(max_retries=5)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROVISION_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content or ""

def answer_question_parser(query: str) -> AgentResponse:
    """ParSeR: 조항 예측 → 다중 검색 → 생성"""
    question_only = _extract_question(query)
    provisions = _predict_provisions(query)

    # 세 가지 관점으로 검색
    docs_by_question = retrieve(question_only, top_k=3)
    docs_by_full = retrieve(query, top_k=3)
    docs_by_provision = retrieve(provisions, top_k=3) if provisions.strip() else []

    # 중복 제거하며 병합 (provision 결과 우선)
    seen = set()
    merged: list[str] = []
    for doc in docs_by_provision + docs_by_full + docs_by_question:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= 7:
            break

    return generate(query, merged)
```

### 결과
- **정확도**: 55.60%
- **실행시간**: 168.6초

---

## Stage 3: Legal Reasoning Prompting

### 아이디어
IRAC(Issue, Rule, Application, Conclusion) 구조를 시스템 프롬프트에 명시하고 Few-shot 예시를 제공하여, LLM이 법률적 추론 과정을 따르도록 유도한다. Stage 2의 ParSeR 검색과 결합.

### 처리 과정
1. Stage 2의 ParSeR 방식으로 문서 검색 (조항 예측 → 3중 검색 → 병합)
2. IRAC 구조를 명시한 시스템 프롬프트 구성
3. 3개의 Few-shot 예시를 대화 히스토리에 추가 (질문-답변 쌍)
4. 검색된 컨텍스트와 질문을 Few-shot 예시 다음에 배치
5. Structured Output으로 정답 파싱

### 핵심 코드

```python
SYSTEM_PROMPT = """\
너는 대한민국 법률 데이터에 기반해 판례와 법리를 완벽하게 분석하는 AI 판사다.
감정을 배제하고, 오직 논리와 증거, 법적 근거에 따라서만 답을 도출하라.

답변 절차 (IRAC):
1. Issue: 문제가 묻는 쟁점을 파악하라.
2. Rule: 해당 쟁점에 적용되는 법률/판례를 떠올려라.
3. Application: 각 선지에 법리를 적용하라.
4. Conclusion: 정답 선지를 결정하라.

아래 참고자료가 주어지면 활용하되, 맹목적으로 따르지 마라.
반드시 A, B, C, D 중 하나만 답해라."""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "예시 문제 1\n문제: 형사소송법상 고소ㆍ고발에 관한 설명..."},
    {"role": "assistant", "content": '{"answer": "D"}'},
    {"role": "user", "content": "예시 문제 2\n문제: 화이트칼라 범죄의 특징..."},
    {"role": "assistant", "content": '{"answer": "C"}'},
    {"role": "user", "content": "예시 문제 3\n문제: 상법상 회사 설립..."},
    {"role": "assistant", "content": '{"answer": "B"}'},
]

def answer_question_legal_reasoning(query: str) -> AgentResponse:
    """Stage 3: ParSeR 검색 + IRAC/Few-shot 생성"""
    client = OpenAI(max_retries=5)

    # --- ParSeR 검색 ---
    provisions = _predict_provisions(client, query)
    # ... (Stage 2와 동일한 3중 검색 및 병합)

    # --- IRAC + Few-shot 생성 ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message},
    ]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=AgentResponse,
        temperature=0.0,
    )
    return response.choices[0].message.parsed
```

### 결과
- **정확도**: 54.83%
- **실행시간**: 204.1초

---

## Stage 4: Log-Likelihood 기반 추론

### 아이디어
LLM에게 자유 형식 답변 대신 각 선택지(A/B/C/D)의 생성 확률(log-likelihood)을 비교하게 하여, 가장 높은 확률의 답을 선택한다. Stage 2의 ParSeR 검색과 결합.

### 이론적 배경
Log-Likelihood 기반 추론은 언어 모델의 토큰 생성 확률 분포를 직접 활용하는 기법이다. 일반적인 생성 방식은 LLM이 자유롭게 텍스트를 생성하지만, 객관식 문제는 답이 A/B/C/D로 제한되어 있다. OpenAI API의 `logprobs` 파라미터를 사용하면 각 토큰의 로그 확률값을 반환받을 수 있으며, 이를 통해 A/B/C/D 토큰의 확률을 직접 비교할 수 있다. 이는 모델이 내부적으로 가장 확신하는 답을 선택하는 방식으로, CoT나 Few-shot에 비해 단일 토큰 생성만으로 답을 결정하므로 토큰 사용량이 적고 일관성이 높다.

### 처리 과정
1. Stage 2의 ParSeR 방식으로 문서 검색
2. 시스템 프롬프트에 "하나의 알파벳만 출력" 지시 추가
3. `max_tokens=1`, `logprobs=True`, `top_logprobs=10` 설정으로 API 호출
4. 응답에서 `logprobs.content[0].top_logprobs` 추출
5. 상위 10개 토큰 중 A/B/C/D에 해당하는 토큰의 로그 확률 비교
6. 가장 높은 로그 확률을 가진 선택지를 정답으로 선택

### 핵심 코드

```python
def answer_question_loglikelihood(query: str) -> AgentResponse:
    """Log-Likelihood: logprobs로 A/B/C/D 확률 비교"""
    client = OpenAI(max_retries=5)

    # --- ParSeR 검색 ---
    docs = _retrieve_with_parser(client, query)

    # --- Log-likelihood 기반 추론 ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
    )

    # logprobs에서 A, B, C, D 확률 추출
    options = {"A", "B", "C", "D"}
    best_answer = "A"
    best_logprob = -math.inf

    choice = response.choices[0]
    if choice.logprobs and choice.logprobs.content:
        top_logprobs = choice.logprobs.content[0].top_logprobs
        for lp in top_logprobs:
            token = lp.token.strip().upper()
            if token in options and lp.logprob > best_logprob:
                best_logprob = lp.logprob
                best_answer = token

    return AgentResponse(answer=best_answer)
```

### 결과
- **정확도**: 54.83%
- **실행시간**: 193.7초

---

## Stage 5: Self-Verification

### 아이디어
초기 답변을 생성한 후, LLM이 자체적으로 해당 답변을 검증하여 오류를 수정하고 최종 답변을 결정하는 2단계 추론 파이프라인.

### 이론적 배경
**Self-Consistency와 Self-Verification**
- Self-Consistency는 동일한 문제에 대해 여러 추론 경로를 생성하고 다수결로 답을 선택하는 방법론 (Wang et al., 2022)
- Self-Verification은 모델이 자신의 답변을 비판적으로 재검토하여 오류를 탐지하고 수정하는 메타인지적 접근
- 법률 도메인에서는 참고 판례와의 충돌 여부, 각 선지의 법리 적합성을 체계적으로 재평가하는 것이 핵심

본 구현은 IRAC(Issue-Rule-Application-Conclusion) 법률 추론 프레임워크를 초기 답변 생성에 적용한 후, 별도의 검증 프롬프트로 참고자료와의 일관성을 재확인하는 이중 검증 구조를 채택.

### 처리 과정
1. **ParSeR 기반 문서 검색**
   - 문제 텍스트에서 관련 법률 조항을 LLM으로 예측
   - 질문만, 전체 쿼리, 예측 조항 각각으로 검색 수행
   - 중복 제거 후 최대 7개 문서 병합

2. **초기 답변 생성 (Stage 3 방식)**
   - IRAC 프레임워크가 포함된 시스템 프롬프트 사용
   - Few-shot 예시 3개 제공
   - 검색된 참고자료와 함께 구조화된 JSON 응답 생성

3. **자기 검증 단계**
   - 초기 답변, 참고자료, 문제를 검증 프롬프트에 입력
   - 법률 문구와의 충돌 여부 확인
   - 각 선지를 재검토하여 최종 답변 결정
   - 오류 발견 시 수정된 답변 반환

### 핵심 코드
```python
def answer_question_self_verification(query: str) -> AgentResponse:
    """Stage 5: 초기 답변 생성 → 자기 검증 → 최종 답변"""
    client = OpenAI(max_retries=5)

    # --- Step 1: 초기 답변 생성 ---
    docs = _retrieve_with_parser(client, query)
    context_block = "\n\n---\n\n".join(
        f"[참고 {i+1}]\n{doc}" for i, doc in enumerate(docs)
    )

    messages_initial = [
        {"role": "system", "content": SYSTEM_PROMPT},  # IRAC 프레임워크
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"## 참고자료\n{context_block}\n\n## 풀어야 할 문제\n{query}"},
    ]

    initial_response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages_initial,
        response_format=AgentResponse,
        temperature=0.0,
    )
    initial_answer = initial_response.choices[0].message.parsed

    # --- Step 2: 자기 검증 ---
    verification_message = (
        f"## 참고자료\n{context_block}\n\n"
        f"## 문제\n{query}\n\n"
        f"## 초기 답변: {initial_answer.answer}\n\n"
        f"위 초기 답변을 검토하고, 최종 답변을 제시하라."
    )

    verified_response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": VERIFICATION_PROMPT},
            {"role": "user", "content": verification_message},
        ],
        response_format=AgentResponse,
        temperature=0.0,
    )

    return verified_response.choices[0].message.parsed
```

**VERIFICATION_PROMPT 핵심 요소:**
```python
VERIFICATION_PROMPT = """\
너는 법률 문제 검토 위원이다.
임무:
1. 초기 답변이 참고자료의 법률 문구와 충돌하지 않는지 확인하라.
2. 각 선지를 하나씩 재검토하여, 초기 답변이 정말 맞는지 판단하라.
3. 만약 초기 답변이 틀렸다면, 올바른 답으로 수정하라.
"""
```

### 결과
- **정확도**: 54.44%
- **실행시간**: 257.3초
- **분석**: Stage 2 ParSeR(55.60%)보다는 낮지만, 이중 추론을 통한 오류 수정 효과가 확인됨. 다만 실행시간이 2배 이상 증가하는 것이 단점.

---

## Stage 6: SAC + Hybrid Dense/BM25 Search

### 아이디어
문서에 분야와 관련법 메타데이터를 접두어로 추가(SAC)하고, Dense 임베딩 검색과 BM25 키워드 검색을 융합하여 법률 용어의 정확한 매칭과 의미적 유사성을 동시에 확보.

### 이론적 배경
**Summary Augmented Chunking (SAC)**
- 각 문서 청크에 요약 정보나 메타데이터를 접두어로 추가하여 검색 정밀도를 높이는 기법
- 본 구현에서는 분야(형법/법학)와 관련 법률명을 추출하여 문서 앞에 구조화된 태그로 삽입
- 검색 시 쿼리가 이러한 메타데이터와 먼저 매칭되어 관련성 높은 문서가 우선 검색됨

**Hybrid Dense + BM25 Search**
- Dense retrieval: 문맥적 의미 유사성을 포착하지만 정확한 키워드 매칭에 약함
- BM25: TF-IDF 기반 키워드 검색으로 전문 법률 용어 정확 매칭에 강점
- Reciprocal Rank Fusion (RRF): 두 검색 결과의 순위를 역수로 변환하여 가중 결합
  - `score = α / (rank_dense + 1) + (1-α) / (rank_bm25 + 1)`
  - α=0.4로 BM25에 더 높은 가중치 부여 (법률 용어 정확성 우선)

### 처리 과정
1. **SAC 문서 인덱싱 (초기화 시 1회)**
   - train.csv에서 모든 문제 읽기
   - 각 문제에서 정규식으로 법률명 추출 (`[가-힣]{2,}(?:기본)?법`)
   - 문서 형식: `[분야: 형법] [관련법: 형사소송법, 형법] [질문] ... [정답]`
   - ChromaDB에 Dense 임베딩 저장, BM25 인덱스 별도 구축

2. **ParSeR 법률 조항 예측**
   - LLM에게 문제를 풀기 위해 필요한 법률명과 조항 예측 요청
   - 예측 결과를 별도 검색 쿼리로 활용

3. **Hybrid 검색 수행**
   - Dense 검색: ChromaDB 벡터 유사도 (top_k×2)
   - BM25 검색: 한국어 토크나이징 후 BM25 스코어링 (top_k×2)
   - RRF로 두 결과 병합, 상위 top_k개 선택
   - 전체 쿼리 검색 + 예측 조항 검색 결과를 중복 제거하여 최대 5개 병합

4. **Few-shot 추론**
   - 검색된 참고자료와 함께 구조화된 프롬프트 생성
   - 3개 Few-shot 예시 포함, JSON 응답 파싱

### 핵심 코드
```python
def _build_sac_document(row: dict[str, str]) -> str:
    """SAC: 분야/관련법 접두어를 추가한 문서 생성."""
    category = row.get("Category", "Law")
    category_kr = "형법" if "Criminal" in category else "법학"

    all_text = row["question"] + " " + " ".join(row[k] for k in "ABCD")
    laws = _extract_laws(all_text)
    law_str = ", ".join(laws[:5]) if laws else "일반"

    answer_label = ANSWER_MAP.get(row["answer"], row["answer"])
    return (
        f"[분야: {category_kr}] [관련법: {law_str}]\n"
        f"[질문] {row['question']}\n"
        f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n"
        f"[정답] {answer_label}"
    )
```

```python
def _hybrid_search(query: str, alpha: float = 0.4, top_k: int = 5) -> list[str]:
    """Hybrid: Reciprocal Rank Fusion of Dense + BM25."""
    dense_docs = _dense_search(query, top_k=top_k * 2)
    bm25_results = _bm25_search(query, top_k=top_k * 2)
    bm25_docs = [doc for doc, _ in bm25_results]

    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense_docs):
        scores[doc] = scores.get(doc, 0) + alpha / (rank + 1)
    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0) + (1 - alpha) / (rank + 1)

    sorted_docs = sorted(scores, key=scores.get, reverse=True)
    return sorted_docs[:top_k]
```

### 결과
- **정확도**: 52.90%
- **실행시간**: 188.3초
- **분석**: SAC 메타데이터 추가와 Hybrid 검색으로 검색 품질이 향상되었으나, Self-Verification 없이는 정확도 54%를 넘지 못함. 실행시간은 단일 추론으로 효율적.

---

## Stage 7: Product of Experts (PoE) + Token Bias Debiasing

### 아이디어
LLM의 선지 알파벳(A/B/C/D) 선호 편향을 측정하여 보정하고, 평균 이하 확률의 선지를 소거한 후 최고 확률 선지를 선택하는 디바이어싱 전략.

### 이론적 배경
**Token Bias in Language Models**
- LLM은 학습 데이터 분포와 토크나이저 특성으로 인해 특정 토큰(예: 'A')을 더 선호하는 경향 존재
- 법률 문제와 무관한 중립적 프롬프트에서 A/B/C/D의 logprob을 측정하여 기본 편향 추정
- Contextual calibration: `debiased_score = raw_logprob - bias_logprob`

**Product of Experts (PoE) Decision Rule**
- 원래 PoE는 여러 전문가 모델의 확률을 곱하는 앙상블 기법
- 본 구현에서는 "평균 이하 확률 선지 배제" 휴리스틱으로 해석
- 4개 선지의 평균 logprob 계산 후, 평균 이상인 선지만 후보로 유지
- 남은 후보 중 최고 debiased logprob 선택

### 처리 과정
1. **Bias Calibration (초기화 시 1회)**
   - 5개의 중립적 객관식 프롬프트 사용 (내용 없는 A/B/C/D 선택)
   - 각 프롬프트에서 A/B/C/D의 logprob 수집
   - 평균하여 각 알파벳의 기본 편향 값 저장

2. **SAC + Hybrid 검색**
   - Stage 6과 동일한 ParSeR + Hybrid 검색 파이프라인
   - 전체 쿼리와 예측 조항으로 검색, 최대 5개 문서 병합

3. **Debiased Logprob 계산**
   - LLM에 참고자료와 문제 입력, max_tokens=1로 A/B/C/D 중 하나만 생성
   - top_logprobs=10으로 A/B/C/D 각각의 logprob 수집
   - 각 선지의 raw logprob에서 calibration bias 차감

4. **PoE 선택 전략**
   - 유효 선지(logprob > -∞)의 평균 계산
   - 평균 이상인 선지만 후보로 유지
   - 후보 중 debiased logprob 최대값 선택

### 핵심 코드
```python
def _calibrate_bias(client: OpenAI) -> dict[str, float]:
    """모델의 A/B/C/D 토큰 선호 편향을 측정한다."""
    global _bias
    if _bias is not None:
        return _bias

    bias = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    count = 0

    for prompt in NEUTRAL_PROMPTS:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
            temperature=0.0,
        )
        choice = resp.choices[0]
        if choice.logprobs and choice.logprobs.content:
            for lp in choice.logprobs.content[0].top_logprobs:
                token = lp.token.strip().upper()
                if token in bias:
                    bias[token] += lp.logprob
            count += 1

    if count > 0:
        for key in bias:
            bias[key] /= count

    _bias = bias
    return _bias
```

```python
def _poe_select(scores: dict[str, float]) -> str:
    """PoE: 평균 이하 선지 소거 후 최고 확률 선택."""
    valid = {k: v for k, v in scores.items() if v > -math.inf}
    if not valid:
        return "A"

    avg = sum(valid.values()) / len(valid)
    candidates = {k: v for k, v in valid.items() if v >= avg}
    if not candidates:
        candidates = valid

    return max(candidates, key=lambda k: candidates[k])
```

### 결과
- **정확도**: 48.65%
- **실행시간**: 179.3초
- **분석**: Debiasing과 PoE 전략이 이론적으로는 편향을 줄이지만, 본 데이터셋에서는 오히려 정확도 하락. 법률 문제의 선지 분포가 균등하지 않거나, Few-shot 학습 없이 logprob만 사용한 것이 원인으로 추정.

---

## Stage 8: Adaptive RAG + CISC (Confidence-weighted Self-Consistency)

### 아이디어
확신도가 높은 문제는 RAG 없이 직접 답변하여 노이즈를 차단하고, 확신도가 낮은 문제만 RAG + 다중 경로 샘플링으로 신뢰도 가중 투표를 수행하는 적응형 라우팅 전략.

### 이론적 배경
**Adaptive RAG Routing**
- 모든 문제에 RAG를 적용하면 관련 없는 문서가 노이즈로 작용할 수 있음
- 모델이 사전 지식만으로 높은 확신도로 답할 수 있는 문제는 RAG 생략
- Confidence threshold (logprob ≥ -0.35, 즉 확률 ≥ 70%)로 라우팅 결정

**CISC: Confidence-weighted In-context Self-Calibration**
- Self-Consistency의 변형: 다수결 대신 각 샘플의 신뢰도를 가중치로 사용
- Temperature > 0으로 N개 샘플 생성 (다양한 추론 경로)
- 각 샘플의 logprob을 확률로 변환하여 가중치 계산
- `vote_weight[answer] += exp(logprob)` 합산 후 최대 가중치 답 선택

### 처리 과정
1. **직접 답변 시도 (No RAG)**
   - Few-shot 프롬프트만으로 문제 입력
   - max_tokens=1, temperature=0으로 즉시 답변 생성
   - A/B/C/D 중 가장 높은 logprob와 해당 답변 추출

2. **Confidence Routing**
   - 신뢰도 ≥ -0.35 (확률 ≥ 70%): 직접 답변 반환
   - 신뢰도 < -0.35: RAG + CISC 경로로 진입

3. **RAG + CISC (낮은 확신 경우)**
   - ParSeR 법률 조항 예측
   - SAC + Hybrid 검색으로 최대 5개 문서 검색
   - Temperature=0.7로 N=3개 샘플 생성
   - 각 샘플의 답변과 logprob 수집
   - `vote_weight[answer] += exp(logprob)` 가중치 누적
   - 최대 가중치 답변 선택

### 핵심 코드
```python
def _get_direct_answer(client: OpenAI, query: str) -> tuple[str, float]:
    """RAG 없이 직접 답변 + 신뢰도(max logprob) 반환."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT,
            {"role": "user", "content": query},
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        temperature=0.0,
    )

    options = {"A", "B", "C", "D"}
    best_answer = "A"
    best_logprob = -math.inf

    choice = resp.choices[0]
    if choice.logprobs and choice.logprobs.content:
        for lp in choice.logprobs.content[0].top_logprobs:
            token = lp.token.strip().upper()
            if token in options and lp.logprob > best_logprob:
                best_logprob = lp.logprob
                best_answer = token

    return best_answer, best_logprob
```

```python
def answer_question_adaptive_cisc(query: str) -> AgentResponse:
    """Stage 8: Adaptive RAG Router + CISC."""
    client = OpenAI(max_retries=5)

    # Step 1: 직접 답변 시도
    direct_answer, confidence = _get_direct_answer(client, query)

    # Step 2: 신뢰도 판단
    if confidence >= CONFIDENCE_THRESHOLD:  # -0.35 (70%)
        return AgentResponse(answer=direct_answer)

    # Step 3: 낮은 확신 → RAG + CISC
    answer = _rag_cisc(client, query)
    return AgentResponse(answer=answer)
```

### 결과
- **정확도**: 47.49%
- **실행시간**: 379.4초
- **분석**: Adaptive 라우팅과 CISC의 조합이 이론적으로는 효율적이나, 실제로는 낮은 정확도와 긴 실행시간(다중 샘플링)으로 성능 저하. Confidence threshold 조정 또는 RAG 없는 직접 답변의 성능이 예상보다 낮았을 가능성.

---

## Stage 9: Chain-of-Thought (CoT) + ParSeR Retrieval

### 아이디어
모델이 단계적으로 추론 과정을 명시적으로 작성한 후 최종 답변을 도출하도록 강제하여, 성급한 오답과 논리적 비약을 방지한다. ParSeR 검색 방식과 결합하여 관련 법률 조항을 먼저 예측하고 이를 기반으로 검색한다.

### 이론적 배경
**Chain-of-Thought (CoT) Prompting** (Wei et al., 2022)은 대형 언어 모델이 중간 추론 단계를 명시적으로 생성하도록 유도하여 복잡한 추론 능력을 향상시키는 기법이다. "Let's think step by step"과 같은 프롬프트를 통해 모델이 최종 답변에 도달하기 전에 단계별 사고 과정을 텍스트로 출력하게 함으로써, 산술 추론, 상식 추론, 기호 추론 등 다양한 벤치마크에서 성능 향상을 보였다. 법률 문제 풀이에서는 각 선지의 법적 정확성을 순차적으로 판단하고 관련 법리를 확인하는 과정이 CoT로 구조화된다.

### 처리 과정
1. **조항 예측 단계**: GPT-4o-mini를 사용하여 입력된 법률 문제로부터 관련 법률명과 조항 번호를 최대 3개까지 예측
2. **다중 검색 수행**: 예측된 조항, 전체 질문, 질문 부분만을 각각 쿼리로 상위 3개 문서씩 검색
3. **문서 병합 및 중복 제거**: provisions → full query → question-only 순서로 최대 7개 문서 선택
4. **CoT 프롬프팅**: 시스템 프롬프트에서 "추론 절차"를 명시 (쟁점 파악 → 각 선지 판단 → 법리 확인 → 최종 결정)
5. **구조화된 응답 생성**: Pydantic `CoTResponse`로 `reasoning`과 `answer` 필드 분리
6. **답변 추출**: 파싱된 응답에서 최종 답변(A/B/C/D)만 반환

### 핵심 코드

```python
class CoTResponse(BaseModel):
    reasoning: str = Field(description="단계적 추론 과정")
    answer: Literal["A", "B", "C", "D"] = Field(description="최종 답변")

SYSTEM_PROMPT = """
너는 대한민국 법률 전문가다.
아래 참고자료와 문제를 읽고, 단계적으로 추론하여 정답을 도출하라.

추론 절차:
1. 문제의 핵심 쟁점을 한 줄로 파악하라.
2. 각 선지(A, B, C, D)의 법적 정확성을 간단히 판단하라.
3. 참고자료에서 관련 법리를 확인하라.
4. 최종 정답을 결정하라.
"""

def answer_question_cot(query: str) -> AgentResponse:
    client = OpenAI(max_retries=5)
    question_only = _extract_question(query)

    # ParSeR Retrieval
    provisions = _predict_provisions(client, query)
    docs_q = retrieve(question_only, top_k=3)
    docs_full = retrieve(query, top_k=3)
    docs_prov = retrieve(provisions, top_k=3) if provisions.strip() else []

    # 병합 및 중복 제거 (최대 7문서)
    seen = set()
    merged: list[str] = []
    for doc in docs_prov + docs_full + docs_q:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= 7:
            break

    # CoT Generation
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=CoTResponse,
        temperature=0.0,
    )

    parsed = response.choices[0].message.parsed
    return AgentResponse(answer=parsed.answer)
```

### 결과
- **정확도**: 52.12%
- **실행 시간**: 484.7초
- **분석**: CoT 방식이 명시적 추론 과정을 유도하지만, 추가 생성 비용으로 인해 실행 시간이 길어졌고, Stage 2 ParSeR(55.60%)나 Stage 3 Legal Reasoning(54.83%)보다 낮은 성능을 보였다. 법률 문제에서는 중간 추론 과정보다 정확한 검색 결과와 간결한 판단이 더 효과적일 수 있음을 시사한다.

---

## Stage 10: Offline Ensemble (Majority Vote)

### 아이디어
기존에 실행한 여러 Stage들의 예측 결과를 다수결 투표로 결합하여 개별 모델의 오류를 상호 보완한다. API 호출 없이 저장된 CSV 파일만 활용하므로 추가 비용이 발생하지 않는다.

### 이론적 배경
**앙상블 학습(Ensemble Learning)**은 여러 개별 모델의 예측을 결합하여 단일 모델보다 더 강건하고 일반화된 예측을 만드는 기법이다. 다수결 투표(Majority Voting)는 가장 단순하면서도 효과적인 앙상블 기법으로, 각 모델이 서로 다른 유형의 오류를 범한다면 상호 보완을 통해 전체 정확도를 향상시킬 수 있다. 법률 QA 시스템에서는 각 Stage가 서로 다른 검색 전략과 프롬프팅 기법을 사용하므로, 특정 유형의 문제에서 강점이 다르다. 이러한 다양성이 앙상블의 핵심 전제 조건이다.

### 처리 과정
1. **예측 결과 로드**: 지정된 Stage들의 CSV 파일에서 `predicted` 컬럼을 읽어온다.
2. **Ground Truth 로드**: 기준이 되는 Stage의 CSV에서 `ground_truth` 컬럼을 로드한다.
3. **다수결 투표**: 각 문제에 대해 모든 Stage의 예측을 모아 `Counter`로 빈도를 계산하고, 가장 많은 표를 받은 답변을 선택한다.
4. **동점 처리**: 최다 득표가 2개 이상인 경우, 미리 지정된 `tiebreak_stage`의 답변을 따른다.
5. **정확도 계산 및 결과 저장**

### 핵심 코드

```python
def run_ensemble(
    stage_names: list[str],
    ensemble_name: str,
    tiebreak_stage: str | None = None,
) -> dict[str, object]:
    """다수결 앙상블 실행"""

    # 각 stage의 예측 로드
    all_predictions: dict[str, list[str]] = {}
    for name in stage_names:
        all_predictions[name] = _load_predictions(name)

    ground_truths = _load_ground_truths(stage_names[0])

    # 다수결 투표
    ensemble_predictions: list[str] = []
    for i in range(len(ground_truths)):
        votes = [all_predictions[name][i] for name in stage_names]
        counter = Counter(votes)
        max_count = counter.most_common(1)[0][1]
        top_candidates = [ans for ans, cnt in counter.items() if cnt == max_count]

        if len(top_candidates) == 1:
            prediction = top_candidates[0]
        else:
            # 동점 → tiebreak_stage의 답 사용
            prediction = all_predictions[tiebreak_stage][i]

        ensemble_predictions.append(prediction)

    return {"accuracy": accuracy, "correct": correct_count, "total": total}
```

### 결과
- **10a_ensemble_top3** (Stages 2, 3, 4): 54.05%, 0.0초
- **10b_ensemble_top5** (Stages 0, 2, 3, 4, 5): 54.44%, 0.0초
- **10c_ensemble_diverse** (Stages 2, 3, 0): **55.21%**, 0.0초
- **분석**: 다양성을 고려한 10c 조합이 가장 높은 성능을 달성. Stage 0(baseline)을 포함한 diverse 앙상블이 top-performing stages만 결합한 것보다 효과적이었다.

---

## Stage 11: Focused Retrieval (Fewer, Better Docs)

### 아이디어
ParSeR 검색 방식을 유지하되, 검색 문서 수를 대폭 축소하여 노이즈를 줄이고 모델이 핵심 정보에 집중하도록 한다.

### 이론적 배경
**Focused Retrieval**은 정보 검색에서 precision을 극대화하기 위해 검색 결과의 개수를 제한하는 접근법이다. RAG 시스템에서 너무 많은 문서를 제공하면 오히려 노이즈가 증가하여 모델이 핵심 정보를 놓칠 수 있다. Stage 0 (5문서, 52.90%)과 Stage 1 (7문서, 50.97%)의 비교에서 문서 수가 적을 때 성능이 더 높았던 관찰에 기반한다.

### 처리 과정
1. **조항 예측**: GPT-4o-mini를 사용하여 관련 법률 조항을 예측
2. **축소된 검색 수행**: 각 쿼리당 `top_k=2`로 검색 (기존 3 → 2)
3. **문서 병합 및 제한**: 최대 4개 문서로 제한 (기존 7 → 4)
4. **생성**: 축소된 문서 집합을 컨텍스트로 제공

### 핵심 코드

```python
def answer_question_focused(query: str) -> AgentResponse:
    """Stage 11: ParSeR + Focused Retrieval (top_k=2, max 4 docs)"""
    client = OpenAI(max_retries=5)
    question_only = _extract_question(query)

    provisions = _predict_provisions(client, query)

    # top_k=2로 축소 (기존 3 → 2)
    docs_prov = retrieve(provisions, top_k=2) if provisions.strip() else []
    docs_full = retrieve(query, top_k=2)
    docs_q = retrieve(question_only, top_k=2)

    # 최대 4문서로 제한 (기존 7 → 4)
    seen = set()
    merged: list[str] = []
    for doc in docs_prov + docs_full + docs_q:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= 4:  # 핵심 변경점
            break

    # Generation
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=AgentResponse,
        temperature=0.0,
    )

    return response.choices[0].message.parsed
```

### 결과
- **정확도**: 53.67%
- **실행 시간**: 302.7초
- **분석**: Stage 2 ParSeR(55.60%)보다는 낮지만, Stage 9 CoT(52.12%)보다 1.55%p 높은 성능. 문서 수 축소가 노이즈 감소에 효과적이나, 극적인 향상은 아니었다.

---

## Stage 12: Dynamic Few-Shot Learning

### 아이디어
고정된 few-shot 예시 대신, 검색된 유사한 문제-답변 쌍을 대화 형식의 동적 few-shot 예시로 변환하여 in-context learning을 극대화한다.

### 이론적 배경
**Dynamic Few-Shot Learning**은 각 입력마다 관련성 높은 예시를 동적으로 선택하여 few-shot 예시로 활용하는 방식이다. GPT 계열 모델은 대화 히스토리(user-assistant 쌍)를 통해 패턴을 학습하는 in-context learning 능력이 뛰어나다. 기존 방식이 검색된 문서를 단일 user message 내에 "참고자료"로 포함시키는 반면, 이 방식은 검색된 상위 문서들을 (question, answer) 쌍으로 변환하여 대화 히스토리에 삽입한다.

### 처리 과정
1. **ParSeR 검색**: 최대 7개 문서 병합
2. **문서 분리**: 상위 3개는 few-shot 예시로, 나머지는 참고자료로 사용
3. **Few-Shot 변환**: 정규표현식으로 `[질문]...[정답]` 형식에서 질문-답변 쌍 추출
4. **대화 히스토리 구성**: `[system] → [few-shot] → [context + question]` 순서

### 핵심 코드

```python
def _parse_retrieved_doc(doc: str) -> tuple[str, str] | None:
    """검색된 문서에서 질문+선지와 정답을 추출"""
    answer_match = re.search(r'\[정답\]\s*([A-D])', doc)
    if not answer_match:
        return None
    answer = answer_match.group(1)

    question_match = re.search(r'\[질문\]\s*(.+?)(?=\[정답\])', doc, re.DOTALL)
    if not question_match:
        return None
    question_text = question_match.group(1).strip()
    question_text = re.sub(r'([A-D])\)', r'\1.', question_text)

    return question_text, answer

def _build_dynamic_fewshot(docs: list[str], max_examples: int = 3) -> list[dict[str, str]]:
    """검색된 문서들을 대화형 few-shot 예시로 변환"""
    fewshot: list[dict[str, str]] = []
    for doc in docs:
        if len(fewshot) >= max_examples * 2:
            break
        parsed = _parse_retrieved_doc(doc)
        if parsed is None:
            continue
        question_text, answer = parsed
        fewshot.append({"role": "user", "content": question_text})
        fewshot.append({"role": "assistant", "content": f'{{"answer": "{answer}"}}'})
    return fewshot
```

### 결과
- **정확도**: 50.58%
- **실행 시간**: 350.3초
- **분석**: 예상과 달리 낮은 성능. 원인: (1) 문서 파싱 오류, (2) 가장 관련성 높은 문서를 few-shot으로 분리하여 상세 참조 불가, (3) 긴 대화 히스토리로 모델 혼동. 법률 QA에서는 "패턴 학습"보다 "구체적 법리 참조"가 더 중요함을 시사.

---

## Stage 13: PRIdE Debiasing + Length Normalization

### 아이디어
GPT-4o-mini의 위치 편향(C 과다예측 ~27%, D 과소예측 ~23%)을 컨텐츠 무관 중립 프롬프트로 측정한 prior 확률로 보정하여 logprobs 기반 디바이어싱을 수행한다.

### 이론적 배경
**PRIdE (Prior Debiasing) 기법** (Zheng et al., 2024)은 언어모델이 특정 선택지에 대해 갖는 사전 편향을 보정하는 방법이다:

1. **Prior Calibration**: 컨텐츠 무관 중립 프롬프트를 사용하여 모델의 선택지별 사전 확률 측정
2. **Debiasing Formula**: `debiased_logprob[x] = raw_logprob[x] - α × prior_logprob[x]`
3. **Alpha Tuning**: Stage 7에서 α=1.0으로 과도하게 보정하여 실패했던 경험을 바탕으로, α=0.5 사용

### 처리 과정
1. **Prior Calibration**: 12개 중립 프롬프트로 A/B/C/D 평균 logprob 측정
2. **ParSeR 검색**: 조항 예측 + 3중 검색 + 최대 7개 문서 병합
3. **Debiased Logprobs**: raw logprob에서 α×prior 차감
4. **최종 선택**: debiased 점수 최대값 선택

### 핵심 코드

```python
def _get_debiased_logprobs(
    client: OpenAI,
    messages: list[dict[str, str]],
    alpha: float = 0.5,
) -> dict[str, float]:
    """실제 질문에 대한 raw logprobs를 얻고 prior로 보정한다."""
    prior = _calibrate_pride_prior(client)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        temperature=0.0,
    )

    raw: dict[str, float] = {"A": -math.inf, "B": -math.inf, "C": -math.inf, "D": -math.inf}
    choice = resp.choices[0]
    if choice.logprobs and choice.logprobs.content:
        for lp in choice.logprobs.content[0].top_logprobs:
            token = lp.token.strip().upper()
            if token in raw:
                raw[token] = max(raw[token], lp.logprob)

    # Apply debiasing
    debiased: dict[str, float] = {}
    for key in raw:
        if raw[key] > -math.inf:
            debiased[key] = raw[key] - alpha * prior.get(key, 0.0)
        else:
            debiased[key] = -math.inf

    return debiased
```

### 결과
- **정확도**: 37.84%
- **실행 시간**: 3447.1초 (~57분)
- **분석**: **실패** (baseline 대비 -17.76%p). α=0.5가 한국어 법률 도메인에 부적합하고, 중립 프롬프트의 편향 패턴이 실제 문제와 상이. gpt-4o-mini의 logprobs 신뢰도 자체가 낮을 가능성.

---

## Stage 14: Binary Task Decomposition

### 아이디어
복합 4지선다 문제를 4개의 독립적인 이진 판단(True/False)으로 분해하여, 위치 편향을 제거하고 부정형 질문의 정답률을 향상시킨다.

### 이론적 배경
**Binary Task Decomposition**은 다중 선택 문제를 여러 개의 이진 분류로 변환하는 기법이다:
- **위치 편향 제거**: 각 선택지를 독립적으로 "참/거짓" 판단하면 순서 효과가 사라짐
- **부정형 질문 특화**: "옳지 않은 것은?" 질문의 경우, P("참")이 가장 낮은 것을 선택
- **Logprobs 활용**: 각 이진 판단에서 "참"/"거짓" 토큰의 logprob를 추출하여 확률 계산

### 핵심 코드

```python
def _evaluate_option_binary(client: OpenAI, option_text: str, context_docs: list[str]) -> float:
    """하나의 선택지가 참일 확률을 logprobs로 계산한다."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": BINARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
    )

    prob_true = 0.0
    prob_false = 0.0
    for token_logprob in top_logprobs:
        token = token_logprob.token
        prob = 2.71828 ** token_logprob.logprob
        if "참" in token or "true" in token.lower():
            prob_true += prob
        elif "거짓" in token or "false" in token.lower():
            prob_false += prob

    total = prob_true + prob_false
    return prob_true / total if total > 0 else 0.5
```

### 결과
- **정확도**: 평가 미완료
- **실행 시간**: 10분 시간제한 초과로 kill됨
- **원인**: 문제당 8회 API 호출 (provision 1 + retrieval 3 + binary 4). 135문제 × 8회 × 5초 = 5,400초 (90분 필요).

---

## Stage 15: Adaptive Routing with Confidence-Weighted Voting

### 아이디어
질문 타입("other"/긍정형/부정형)을 자동 분류하여 특화된 파이프라인으로 라우팅하고, 각 파이프라인 내에서 신뢰도 기반 투표로 최종 답변을 결정한다.

### 이론적 배경
**Adaptive Routing**은 입력 데이터의 특성에 따라 최적의 처리 파이프라인을 선택하는 기법이다:

| 질문 타입 | 파이프라인 | 이유 |
|---------|-----------|------|
| Other | IRAC Legal Reasoning + Few-shot | 법률 추론 방법론으로 사실관계 분석에 강점 |
| Negative | Logprobs + Anti-bias Prompt | "틀린 것 찾기"를 명시적으로 프롬프트에 강조 |
| Positive | Standard ParSeR + Structured Output | 안정적인 검색 + 구조화된 출력 |

Primary 파이프라인의 confidence가 threshold 미만이면 Backup 파이프라인을 추가 실행하여 두 결과를 비교한다.

### 핵심 코드

```python
def detect_question_type(query: str) -> str:
    question_text = query.split("\n\n")[0] if "\n\n" in query else query
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in question_text:
            return "negative"
    for keyword in POSITIVE_KEYWORDS:
        if keyword in question_text:
            return "positive"
    return "other"

def answer_question_adaptive(query: str) -> AgentResponse:
    client = OpenAI(max_retries=5)
    question_type = detect_question_type(query)
    context_docs = parser_retrieve(client, query, top_k=7)

    if question_type == "other":
        answer = confidence_weighted_vote(client, query, context_docs, "irac")
    elif question_type == "negative":
        answer = confidence_weighted_vote(client, query, context_docs, "negative_logprobs")
    else:
        answer = confidence_weighted_vote(client, query, context_docs, "positive_structured")

    return AgentResponse(answer=answer)
```

### 결과
- **정확도**: 평가 미완료 (10분 시간제한 초과)
- **원인**: 문제당 7-8회 API 호출 (primary + backup + IRAC few-shot 오버헤드)

---

## Stage 16: Negative-Aware CoT Hybrid (Two-Pass)

### 아이디어
ParSeR의 안정성과 CoT의 독창적 추론 능력을 2-pass 방식으로 결합하되, 부정형 질문에 특화된 CoT 프롬프트를 사용하여 CoT의 약점을 보완한다.

### 이론적 배경
**실험적 발견** (Stage 9 CoT 분석):
- CoT가 다른 모든 방법이 틀린 9개 문제를 유일하게 맞춤
- 사실형(factual) 질문: CoT 80.5% vs ParSeR 55.6%
- 부정형 질문: CoT 45.5% (취약) vs ParSeR 55.6%

**Two-Pass Hybrid 설계**:
```
Pass 1 (ParSeR): 안정적 baseline + confidence 측정
Pass 2 (Negative-Aware CoT): 부정형 질문 보완 + 독창적 추론

Decision Logic:
- 두 답변이 일치 → 그 답변 사용 (높은 신뢰)
- 불일치 + ParSeR confidence ≥ -0.3 (74%) → ParSeR 신뢰
- 불일치 + ParSeR confidence < -0.3 → CoT 채택 (엣지 케이스 가능성)
```

### 핵심 코드

```python
def _get_cot_system_prompt(is_negative: bool) -> str:
    if is_negative:
        return """\
너는 대한민국 법률 전문가다.
이 문제는 법적으로 틀린 선지를 찾는 문제다.
각 선지(A, B, C, D)가 법적으로 참인지 거짓인지 하나씩 분석하라.
거짓인 선지가 정답이다."""
    else:
        return """\
너는 대한민국 법률 전문가다.
각 선지를 하나씩 분석하고 법적 근거를 제시하라.
분석 후 최종 답변을 제시하라."""

def answer_question_neg_cot(query: str) -> AgentResponse:
    # Pass 1: ParSeR Standard
    answer1 = parser_response.choices[0].message.parsed.answer
    confidence1 = extract_confidence(...)

    # Pass 2: Negative-Aware CoT
    is_negative = _is_negative_question(query)
    cot_system_prompt = _get_cot_system_prompt(is_negative)
    answer2 = cot_response.choices[0].message.parsed.answer

    # Decision Logic
    if answer1 == answer2:
        final_answer = answer1
    elif confidence1 >= -0.3:
        final_answer = answer1  # Trust ParSeR
    else:
        final_answer = answer2  # Use CoT's unique insight

    return AgentResponse(answer=final_answer)
```

### 결과
- **정확도**: 평가 미완료 (10분 시간제한 초과)
- **원인**: 문제당 8회 API 호출 (2-pass + confidence extraction + CoT reasoning 생성 오버헤드)

---

## Stage 17: Negative-Aware Prompt ParSeR

### 아이디어
부정형 문제("옳지 않은 것", "틀린 것" 등)를 감지하여 전용 시스템 프롬프트를 적용함으로써 답변 정확도를 향상시킨다. Stage 2의 ParSeR 파이프라인을 유지하되, 프롬프트만 동적으로 교체하여 추가 API 호출 없이 개선을 달성한다.

### 이론적 배경
법률 시험 문제는 크게 두 가지 유형으로 구분된다: (1) "옳은 것을 고르시오"와 (2) "옳지 않은 것을 고르시오". 후자의 경우 LLM이 각 선지의 진위를 명시적으로 판단한 뒤 거짓인 선지를 선택해야 하나, 일반적인 프롬프트에서는 이러한 논리 반전이 명확히 지시되지 않아 혼란을 야기할 수 있다. 키워드 기반 감지를 통해 문제 유형을 식별하고, 부정형 문제에 특화된 프롬프트를 제공하는 기초적인 접근법이다.

### 처리 과정
1. **키워드 기반 문제 유형 감지**: 9개의 부정형 키워드 존재 여부 확인
2. **ParSeR 검색 수행**: Stage 2와 동일
3. **동적 프롬프트 선택**: 부정형이면 `SYSTEM_PROMPT_NEGATIVE`, 그 외 `SYSTEM_PROMPT_DEFAULT`
4. **구조화된 출력 생성**

### 핵심 코드
```python
NEGATIVE_KEYWORDS = [
    "옳지 않은", "적절하지 않은", "해당하지 않는", "아닌 것",
    "틀린 것", "잘못된", "부적절한", "타당하지 않은", "맞지 않는",
]

def _is_negative(query: str) -> bool:
    return any(kw in query for kw in NEGATIVE_KEYWORDS)

def answer_question_neg_prompt(query: str) -> AgentResponse:
    # ParSeR retrieval (Stage 2와 동일)
    # ...
    system_prompt = SYSTEM_PROMPT_NEGATIVE if _is_negative(query) else SYSTEM_PROMPT_DEFAULT

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=AgentResponse,
        temperature=0.0,
    )
```

**부정형 프롬프트 핵심 지시사항**:
```
**주의: 이 문제는 '옳지 않은 것' 또는 '적절하지 않은 것'을 찾는 문제다.**

풀이 방법:
1. 각 선지(A, B, C, D)가 법적으로 올바른 진술인지 하나씩 판단하라.
2. 법적으로 틀린 진술(거짓인 선지)을 정답으로 골라라.
3. 대부분의 선지가 참이고, 하나만 거짓이다.
```

### 결과
- **정확도**: 52.51%
- **실행시간**: 172.0초
- **분석**: Stage 2 ParSeR(55.60%) 대비 소폭 하락. 부정형 전용 프롬프트가 특정 문제에서는 효과적이나, 전체적으로는 기본 프롬프트가 더 나은 결과를 보임.

---

## Stage 18: Anti-Position-Bias Prompt ParSeR

### 아이디어
LLM이 선지의 위치(A/B/C/D)에 편향되지 않도록 시스템 프롬프트에 위치 편향 경고를 추가하여, 각 선지를 동등한 확률로 검토하고 내용의 법적 정확성만으로 판단하게 한다.

### 이론적 배경
**Position Bias**는 LLM이 특정 위치의 선지를 과도하게 선호하거나 회피하는 현상을 의미한다. Zheng et al. (2023)의 연구에 따르면, 다중 선택형 문제에서 LLM은 중간 위치(예: C)를 과도하게 선택하거나, 마지막 위치(D)를 과소평가하는 경향이 있다. 이는 학습 데이터의 불균형, 순서 효과(recency/primacy effect), 또는 모델 내부의 확률 분포 왜곡에서 기인한다.

### 핵심 코드
```python
SYSTEM_PROMPT = """\
너는 대한민국 법률 전문가다.
아래에 참고할 수 있는 유사 기출문제와 정답이 주어진다.
이 참고자료를 활용하여 새로운 문제의 정답을 골라라.

규칙:
- 반드시 A, B, C, D 중 하나만 답해라.
- 참고자료의 패턴과 법리를 활용하되, 맹목적으로 따르지 마라.
- 논리적으로 사고한 뒤 최종 답을 결정해라.

**중요 주의사항:**
- 선지의 위치(A/B/C/D)에 편향되지 말고, 오직 내용의 법적 정확성만으로 판단하라.
- C를 과도하게 선호하거나 D를 과소평가하지 마라.
- 모든 선지를 동등한 확률로 검토한 뒤 답을 결정하라."""
```

### 결과
- **정확도**: 55.21%
- **실행시간**: 177.1초
- **분석**: Stage 2 ParSeR(55.60%)와 거의 동일한 수준. 위치 편향 경고가 일부 문제에서 효과적이었으나, 유의미한 차이는 아님.

---

## Stage 19: Combined Negative-Aware + Anti-Bias ParSeR

### 아이디어
Stage 17의 부정형 전용 프롬프트와 Stage 18의 위치 편향 경고를 결합하여, 문제 유형에 따라 최적화된 프롬프트를 제공하면서도 모든 경우에 위치 편향을 완화한다.

### 이론적 배경
Stage 17과 Stage 18은 각각 독립적인 개선 전략을 제시한다: 전자는 부정형 문제의 논리 반전을 명시화하고, 후자는 위치 편향을 완화한다. 이 두 전략은 상호 배타적이지 않으므로, 동일한 파이프라인에서 결합하여 시너지 효과를 기대할 수 있다.

### 핵심 코드
```python
SYSTEM_PROMPT_DEFAULT = """\
너는 대한민국 법률 전문가다.
...
규칙:
- 선지의 위치(A/B/C/D)에 편향되지 말고, 오직 내용의 법적 정확성만으로 판단하라.
- 모든 선지를 동등한 확률로 검토하라."""

SYSTEM_PROMPT_NEGATIVE = """\
**주의: 이 문제는 '옳지 않은 것' 또는 '적절하지 않은 것'을 찾는 문제다.**
...
규칙:
- 선지의 위치(A/B/C/D)에 편향되지 말고, 오직 내용의 법적 정확성만으로 판단하라.
- 모든 선지를 동등한 확률로 검토하라."""

def answer_question_combined(query: str) -> AgentResponse:
    system_prompt = SYSTEM_PROMPT_NEGATIVE if _is_negative(query) else SYSTEM_PROMPT_DEFAULT
    response = client.beta.chat.completions.parse(...)
```

### 결과
- **정확도**: 54.44%
- **실행시간**: 170.1초
- **분석**: Stage 18(55.21%)보다 약 0.8%p 하락. 부정형 프롬프트 추가가 일부 문제에서 오히려 혼란을 야기. 두 전략의 단순 결합이 항상 시너지를 내지는 않음을 시사.

---

## Stage 20: Smart Offline Ensembles

### 아이디어
기존 실험 결과(Stage 0~9)를 활용하여 API 호출 없이 즉시 실행 가능한 3가지 오프라인 앙상블 전략: (20a) 문제 유형별 가중 투표, (20b) 유형별 최적 방법론 라우팅, (20c) 다양성 기반 앙상블.

### 이론적 배경
**앙상블 학습(Ensemble Learning)**은 여러 모델의 예측을 결합하여 단일 모델보다 높은 성능을 달성하는 기법이다:

1. **Type-Weighted Voting (20a)**: 문제 유형에 따라 각 방법론에 차등 가중치를 부여하여 투표. 도메인 지식 기반의 휴리스틱 앙상블.

2. **Best-per-Type Routing (20b)**: 개발 세트에서 각 문제 유형별로 가장 높은 정확도를 보인 방법론을 식별하고, 해당 유형의 문제는 항상 그 방법론의 예측을 사용. 오라클 기반 라우팅이나 과적합 위험.

3. **Diverse Ensemble (20c)**: 예측 상관관계가 낮은 3개 방법론(ParSeR, CoT, Self-Verification)을 선택하여 다수결 투표. 다양성 기반으로 강건성이 높다.

### 핵심 코드

#### 문제 유형 감지
```python
NEGATIVE_KEYWORDS = [
    "옳지 않은", "적절하지 않은", "해당하지 않는", "아닌 것",
    "틀린 것", "잘못된", "부적절한", "타당하지 않은", "맞지 않는",
]
POSITIVE_KEYWORDS = ["옳은 것", "맞는 것", "적절한 것", "해당하는 것", "타당한 것"]

def _detect_type(question: str) -> str:
    for kw in NEGATIVE_KEYWORDS:
        if kw in question:
            return "negative"
    for kw in POSITIVE_KEYWORDS:
        if kw in question:
            return "positive"
    return "other"
```

#### 20b: Best-per-Type Routing
```python
def run_20b_best_per_type():
    # 유형별로 각 방법론의 정답률 계산
    type_indices: dict[str, list[int]] = {"negative": [], "positive": [], "other": []}
    for i, row in enumerate(dev):
        qtype = _detect_type(row["question"])
        type_indices[qtype].append(i)

    best_stage_per_type: dict[str, str] = {}
    for qtype, indices in type_indices.items():
        best_acc = -1.0
        best_stage = "2_parser"
        for stage in stages:
            correct = sum(1 for i in indices if all_preds[stage][i] == ground_truths[i])
            acc = correct / len(indices) if indices else 0
            if acc > best_acc:
                best_acc = acc
                best_stage = stage
        best_stage_per_type[qtype] = best_stage

    # 라우팅
    predictions = []
    for i, row in enumerate(dev):
        qtype = _detect_type(row["question"])
        predictions.append(all_preds[best_stage_per_type[qtype]][i])
```

### 결과

| 전략 | 정확도 | 실행시간 | 비고 |
|------|--------|----------|------|
| 20a: Type-Weighted Voting | 54.44% | 0.0s | 휴리스틱 가중치 |
| **20b: Best-per-Type Routing** | **57.92%** | **0.0s** | **전체 실험 최고 성능** |
| 20c: Diverse Ensemble | 55.60% | 0.0s | 다양성 기반 |

**20b 유형별 최적 방법론**:
- **부정형 → self_verification**: 52.7% (87/165)
- **긍정형 → loglikelihood**: 56.6% (30/53)
- **기타 → cot**: 80.5% (33/41)

**오버피팅 분석**: 20b는 dev set 기준 57.92%이나, test set(비중복 243문항)에서는 53.50%로 하락. 단일 방법론 loglikelihood(53.09%)와 큰 차이 없음. dev set 기반 최적화가 일반화되지 않았음을 입증.

---

## 종합 분석

### 핵심 발견

**1. 검색(Retrieval) 품질이 가장 중요하다**
- Stage 2 ParSeR(조항 예측 → 3중 검색)이 가장 효과적인 단일 전략 (55.60%)
- 검색 문서 수를 늘리는 것(Stage 1: 7문서 → 50.97%)보다 정확한 검색(Stage 2: 7문서 → 55.60%)이 핵심
- Hybrid 검색(Stage 6)이나 SAC 메타데이터 추가는 큰 개선을 가져오지 못함

**2. 복잡한 후처리는 오히려 해로울 수 있다**
- Debiasing (Stage 7: 48.65%, Stage 13: 37.84%): 편향 보정이 오히려 성능 저하
- Multi-pass (Stage 8: 47.49%): 다중 추론 경로가 일관성을 해침
- 단순한 logprobs 비교(Stage 4: 54.83%)가 복잡한 전략보다 효과적

**3. 프롬프트 엔지니어링은 제한적 효과**
- Stage 17 (부정형 프롬프트): 52.51% (Stage 2보다 하락)
- Stage 18 (위치 편향 경고): 55.21% (Stage 2와 동일 수준)
- Stage 19 (결합): 54.44% (두 전략의 시너지 없음)

**4. 앙상블은 과적합 위험이 있다**
- Stage 20b (dev 최적 라우팅): dev 57.92% → test 53.50% (과적합 입증)
- 단일 방법론 Stage 4: dev 54.83% → test 53.09% (가장 안정적)

**5. 실행시간 제약이 설계를 지배한다**
- 10분 제한 내 완료 필요 (259문항 기준)
- Stage 14-16: 문제당 8회 API 호출 → 시간 초과 실패
- 문제당 API 호출은 2-3회가 실무적 한계

### 방법론 분류

| 카테고리 | Stages | 결과 |
|----------|--------|------|
| **효과적** | 2 (ParSeR), 4 (Loglikelihood), 3 (IRAC) | 54-56% |
| **보통** | 0 (Baseline), 5 (Self-Verify), 9 (CoT) | 52-54% |
| **비효과적** | 1 (OADR), 7 (PoE), 8 (CISC), 12 (Dynamic FS) | 47-51% |
| **실패** | 13 (PRIdE), 14-16 (시간 초과) | <38% / N/A |
| **프롬프트 변형** | 17, 18, 19 | 52-55% (Stage 2와 유사) |
| **앙상블** | 10, 20 | 54-58% (과적합 주의) |

### 최종 결론

**Stage 4 (Log-Likelihood)를 최종 제출 방법론으로 선택.** dev/test 간 가장 일관된 성능(54.83% → 53.09%)을 보이며, 과적합 없이 안정적인 추론을 제공한다. ParSeR 검색으로 관련 문서를 확보하고, logprobs로 모델의 내부 확신도를 직접 비교하는 단순하고 견고한 전략이다.

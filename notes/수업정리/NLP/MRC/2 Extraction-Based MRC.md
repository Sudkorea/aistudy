![[[MRC] (2강) Extraction-based MRC.pdf]]

# Extraction-based MRC

## 정의

Extraction-based MRC(Machine Reading Comprehension)는 질문에 대한 답변이 항상 주어진 지문(context) 내에 존재하는 기계 독해 방식임. 모델은 질문을 이해하고 지문 내에서 정답을 포함하는 텍스트 범위(span)를 찾아내는 것이 목표임.

주요 특징:

- 답변은 항상 지문의 연속된 부분(span)으로 존재함
- 생성이 아닌 추출 방식으로 답변을 찾음
- 지문 내의 정확한 위치(시작과 끝 위치)를 예측함

대표적인 데이터셋:

- SQuAD (Stanford Question Answering Dataset)
- KorQuAD (Korean Question Answering Dataset)
- NewsQA
- TriviaQA

## 평가 방법

### EM Score (Exact Match)

예측한 답변이 정답과 정확히 일치하는지 평가하는 지표임. 완전히 일치하면 1점, 그렇지 않으면 0점을 부여함.

**수식**: $EM = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(prediction_i = answer_i)$

여기서:

- $N$은 전체 질문 수
- $\mathbf{1}$은 지시함수로, 예측값과 정답이 정확히 같을 때 1, 다를 때 0을 반환함

특징:

- 엄격한 평가 방식으로, 사소한 차이(공백, 구두점)에도 0점 처리됨
- 동일한 의미라도 표현이 다르면 불일치로 판단됨
- 평가가 단순하고 직관적임

### F1 Score

예측값과 정답 간의 단어 중복(overlap) 비율로 계산하는 지표임. 정밀도(Precision)와 재현율(Recall)의 조화 평균으로 나타냄.

**수식**: $Precision = \frac{|Prediction \cap Answer|}{|Prediction|}$ $Recall = \frac{|Prediction \cap Answer|}{|Answer|}$ $F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$

여기서:

- $|Prediction \cap Answer|$는 예측과 정답에 공통으로 포함된 단어 수
- $|Prediction|$은 예측한 답변에 포함된 단어 수
- $|Answer|$는 정답에 포함된 단어 수

특징:

- EM보다 유연한 평가 방식으로, 부분적인 일치도 점수에 반영함
- 예측이 정답의 일부만 포함하거나 불필요한 단어를 포함하는 경우도 평가 가능함
- 여러 가능한 정답이 있는 경우 유용함

# Pre-Processing

## Tokenization

텍스트를 작은 단위(토큰)로 나누는 과정임. 딥러닝 모델은 텍스트를 직접 처리할 수 없으므로, 숫자로 변환하기 위한 첫 단계임.

Extraction-based MRC에서의 토크나이징 중요성:

- 토큰 단위가 답변 추출의 기본 단위가 됨
- 토큰화 방식에 따라 정확한 span 위치 예측 성능이 달라질 수 있음
- 토큰화 과정에서 정보 손실이 발생할 수 있음

주요 토크나이저:

- WordPiece (BERT)
- BPE (GPT 계열)
- SentencePiece (다국어 모델)

```python
# BERT 토크나이저 사용 예시
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("What is extraction-based MRC?")
# ['what', 'is', 'extraction', '-', 'based', 'mrc', '?']
```

## Special Tokens

특별한 의미를 가진 토큰으로, 모델에게 입력 구조에 대한 정보를 제공함.

주요 특수 토큰:

- **[CLS]**: 시퀀스의 시작을 나타내며, 문장 분류 태스크에서 전체 시퀀스의 표현으로 사용됨
- **[SEP]**: 두 시퀀스 사이의 구분을 나타냄 (MRC에서는 질문과 지문 사이)
- **[PAD]**: 배치 처리를 위해 시퀀스 길이를 맞추는 패딩 토큰
- **[UNK]**: 어휘집에 없는 미등록 단어(Out-of-Vocabulary)를 나타냄
- **[MASK]**: 마스킹된 단어(BERT의 MLM 학습에서 사용)

MRC에서의 입력 구성:

```
[CLS] 질문 토큰들 [SEP] 지문 토큰들 [SEP]
```

## Attention Mask

입력 시퀀스 중 attention 연산 시 무시할 토큰을 표시하는 이진 마스크임. 주로 패딩 토큰을 무시하기 위해 사용됨.

구성:

- 실제 토큰: 1 (attention 계산에 포함)
- 패딩 토큰: 0 (attention 계산에서 제외)

```python
# attention mask 예시
input_ids = [101, 2054, 2003, 19081, 1010, 2339, 17662, 136, 102, ...]  # 토큰 ID
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, ..., 0, 0, 0]  # 실제 토큰은 1, 패딩은 0
```

중요성:

- 가변 길이 시퀀스 처리 가능
- 효율적인 계산 (불필요한 패딩 토큰 처리 방지)
- 패딩 토큰이 모델의 표현 학습에 영향을 주는 것 방지

## Token Type IDs

입력이 2개 이상의 시퀀스(예: 질문과 지문)로 구성될 때, 각 토큰이 어느 시퀀스에 속하는지 구분하는 ID값임.

구성:

- 첫 번째 시퀀스(질문): 0
- 두 번째 시퀀스(지문): 1

```python
# token type ids 예시 (BERT)
input_text = "[CLS] 질문 [SEP] 지문 [SEP]"
token_type_ids = [0, 0, 0, 0, 1, 1, 1]  # 질문은 0, 지문은 1
```

목적:

- 모델이 질문과 지문을 구분할 수 있게 함
- 상호 참조(cross-reference) 관계 학습에 중요
- 토큰의 문맥적 위치 정보 제공

## 모델 출력값

Extraction-based MRC에서 정답은 지문 내에 존재하는 연속된 단어 토큰(span)이므로, span의 시작과 끝 위치만 파악하면 됨.

주요 특징:

- 답안 생성이 아닌, 시작과 끝 토큰 위치를 예측하는 방식
- Token-Classification 문제로 치환됨
- 각 토큰이 답변의 시작 또는 끝일 확률을 계산

출력 형태:

- **start_logits**: 각 토큰이 답변의 시작일 확률 점수
- **end_logits**: 각 토큰이 답변의 끝일 확률 점수

```python
# 모델 출력 예시
outputs = model(input_ids, attention_mask, token_type_ids)
start_logits = outputs.start_logits  # shape: [batch_size, seq_length]
end_logits = outputs.end_logits      # shape: [batch_size, seq_length]

# 가장 높은 확률의 시작과 끝 위치 선택
start_idx = torch.argmax(start_logits, dim=1)
end_idx = torch.argmax(end_logits, dim=1)
```

# Fine-Tuning

## BERT 기반 MRC 모델

BERT(Bidirectional Encoder Representations from Transformers)는 Extraction-based MRC에 널리 사용되는 모델임. 양방향 문맥 이해 능력이 뛰어나 질문과 지문의 관계를 효과적으로 파악함.

BERT MRC 모델 구조:

1. **입력 임베딩**: 토큰, 위치, 세그먼트 임베딩의 합
2. **인코더 레이어**: 다중 트랜스포머 블록으로 구성
3. **MRC 출력 레이어**: 시작/끝 위치 예측을 위한 두 개의 선형 레이어

```python
# BERT 기반 MRC 모델 fine-tuning 예시
from transformers import BertForQuestionAnswering, AdamW

# 모델 초기화
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=3e-5)

# 학습 루프
for batch in train_dataloader:
    # 입력값 준비
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    start_positions = batch['start_positions']
    end_positions = batch['end_positions']
    
    # 모델 출력 및 손실 계산
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        start_positions=start_positions,
        end_positions=end_positions
    )
    
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Fine-tuning 전략:

- 적절한 학습률 선택 (일반적으로 2e-5 ~ 5e-5)
- 배치 크기 조정 (메모리 제약에 따라 8~32)
- 학습 에폭 설정 (보통 2~4 에폭)
- 가중치 감쇠(weight decay) 적용
- 학습률 스케줄링 (선형 감소 등)

발전된 모델:

- **RoBERTa**: BERT의 개선 버전으로, 더 많은 데이터와 최적화된 학습 방식 적용
- **ALBERT**: 파라미터 공유를 통해 경량화된 BERT
- **ELECTRA**: 더 효율적인 사전 학습 방식을 사용한 모델
- **DeBERTa**: 향상된 디코딩 구조와 상대적 위치 인코딩을 적용한 모델

# Post-processing

## 불가능한 답 제거

모델의 예측 중 논리적으로 불가능한 답변을 필터링하는 과정임. 다음과 같은 경우를 제거함:

1. **end position이 start position보다 앞에 있는 경우**:
    
    - 유효한 span은 시작 위치가 끝 위치보다 앞에 있어야 함
    - `if end_idx < start_idx: continue`
2. **예측한 위치가 context를 벗어난 경우**:
    
    - 답변은 반드시 지문 내에 있어야 함 (질문이나 특수 토큰이 아님)
    - `if start_idx < context_start_idx or end_idx > context_end_idx: continue`
3. **미리 설정한 max length를 벗어난 경우**:
    
    - 지나치게 긴 답변은 일반적으로 잘못된 예측일 가능성이 높음
    - `if end_idx - start_idx + 1 > max_answer_length: continue`

```python
# 불가능한 답 제거 예시
max_answer_length = 30
context_start_idx = question_tokens.index('[SEP]') + 1

valid_predictions = []
for start_idx, end_idx in zip(start_indices, end_indices):
    # 1. 끝 위치가 시작 위치보다 앞에 있는 경우
    if end_idx < start_idx:
        continue
        
    # 2. 예측이 context를 벗어난 경우
    if start_idx < context_start_idx or end_idx >= len(tokens):
        continue
        
    # 3. 최대 길이 초과
    if end_idx - start_idx + 1 > max_answer_length:
        continue
        
    valid_predictions.append((start_idx, end_idx))
```

## 최적의 답안 찾기

여러 가능한 예측 중에서 가장 신뢰도 높은 답변을 선택하는 과정임.

기본 접근법:

1. **최대 점수 전략**: 시작 로짓과 끝 로짓의 합이 최대가 되는 조합 선택
    
    ```python
    # 시작과 끝 로짓 합산 점수 기준으로 최적의 답 선택
    best_score = float('-inf')
    best_start, best_end = 0, 0
    
    for start_idx in range(context_start_idx, len(tokens)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(tokens))):
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start, best_end = start_idx, end_idx
    ```
    
2. **빔 서치(Beam Search)**: 여러 유망한 후보를 동시에 추적하는 방식
    
    ```python
    # N-best 후보 유지
    n_best = 20
    top_start_indices = torch.topk(start_logits, n_best).indices
    top_end_indices = torch.topk(end_logits, n_best).indices
    
    candidates = []
    for start_idx in top_start_indices:
        for end_idx in top_end_indices:
            if end_idx >= start_idx and end_idx - start_idx < max_answer_length:
                score = start_logits[start_idx] + end_logits[end_idx]
                candidates.append((start_idx, end_idx, score))
    
    # 점수 기준 정렬
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_start, best_end = candidates[0][0], candidates[0][1]
    ```
    
3. **앙상블 기법**: 여러 모델의 예측을 결합하여 더 강건한 결과 도출
    
    ```python
    # 여러 모델의 예측 결합
    ensemble_predictions = {}
    for model_idx, model_outputs in enumerate(all_model_outputs):
        start_logits = model_outputs.start_logits
        end_logits = model_outputs.end_logits
        
        # 각 모델의 top k 예측 추출
        for start_idx, end_idx, score in get_top_predictions(start_logits, end_logits, k=5):
            span = (start_idx, end_idx)
            if span not in ensemble_predictions:
                ensemble_predictions[span] = 0
            ensemble_predictions[span] += score
    
    # 결합된 점수로 최종 예측 선택
    best_span = max(ensemble_predictions.items(), key=lambda x: x[1])[0]
    ```
    

추가 고려사항:

- **답변 없음(No Answer) 처리**: SQuAD 2.0과 같은 데이터셋에서는 답이 없는 경우도 처리해야 함
- **중복 답변 병합**: 의미적으로 유사한 여러 답변을 통합하는 방법 (NMS 등)
- **후처리 휴리스틱**: 불완전한 단어나 구두점 처리, 특정 패턴에 기반한 조정 등

Extraction-based MRC는 모델 자체의 성능뿐 아니라, 이러한 전처리 및 후처리 단계의 최적화가 종합적인 성능 향상에 중요한 역할을 함.
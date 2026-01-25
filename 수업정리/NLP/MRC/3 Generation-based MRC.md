![[[MRC] (3강) Generation-based MRC.pdf]]

# Generation-based MRC 
## 정의
주어진 지문과 질의를 보고, 답변을 생성함.(generation)

## 평가방식
앞이랑 똑같은듯?
## Generation-Based Model vs Extraction-based MRC
1) 모델 구조
   Seq-to-seq PLM 구조 vs PLM + Classifier 구조
2) Loss 계산을 위한 답의 형태 / Prediction의 형태
   Free-form text 형태 vs 지문 내 답의 위치(extraction)
# Pre-processing 
앞서 한 Extraction-based MRC랑 거의 비슷한듯? 근데, BERT와 BART 차이때문에 미묘하게 좀 다름

## BERT vs BART
### BERT (Bidirectional Encoder Representations from Transformers)

BERT는 2018년 Google에서 개발한 사전 학습 언어 모델입니다.

- **구조**: 트랜스포머의 인코더 부분만 사용한 양방향 모델
- **사전 학습 방법**:
    - Masked Language Modeling (MLM): 문장의 일부 토큰을 마스킹하고 해당 토큰을 예측
    - Next Sentence Prediction (NSP): 두 문장이 연속적인지 아닌지 예측
- **장점**: 문맥을 양방향으로 이해하여 풍부한 언어 표현을 학습

## BART (Bidirectional and Auto-Regressive Transformers)

BART는 2019년 Facebook AI에서 개발한 시퀀스-투-시퀀스 모델입니다.

- **구조**: 트랜스포머의 인코더-디코더 구조를 모두 사용
- **사전 학습 방법**:
    - 다양한 노이즈 함수(토큰 마스킹, 문장 순서 변경, 토큰 삭제 등)를 적용한 후
    - 원본 텍스트를 복원하는 방식으로 학습
- **장점**: 텍스트 생성 능력이 뛰어나며 요약, 번역 등 생성 작업에 적합

## BERT와 BART의 주요 차이점

1. **구조적 차이**:
    - BERT: 인코더만 사용 (양방향)
    - BART: 인코더-디코더 구조 모두 사용 (양방향 인코더 + 단방향 디코더)
2. **용도 차이**:
    - BERT: 주로 분류, 추출 기반 작업에 적합
    - BART: 텍스트 생성, 요약, 번역 등 생성 작업에 적합
3. **사전 학습 방식 차이**:
    - BERT: MLM과 NSP 두 가지 방식
    - BART: 다양한 노이즈 전략을 통한 텍스트 복원 학습

## 입력 표현 - Special Token
Padd

# Model 
# Post-processing
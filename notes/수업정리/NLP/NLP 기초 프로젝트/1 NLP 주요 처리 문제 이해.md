![[[NLP 기초프로젝트] (1강) 자연어처리 주요문제 이해.pdf]]

# N21 problem
Encoder : 데이터 읽는거
Decoder : 생성하는거
Seq2Seq
n:m 비율에 따라 - 이거 저번 강의에서 들었음

Encoder | Decoder에 대해 개념이 좀 헷갈림

# 도와줘요 클로드선생님
### Encoder-only 모델
- **목적**: 주로 텍스트 이해, 분류, 감성 분석 등 입력 텍스트를 이해하는 작업
- **예시**: BERT, RoBERTa
- **특징**: 양방향으로 문맥을 파악할 수 있음 (문장의 앞뒤 단어를 모두 고려)
- **작동 방식**: 입력 텍스트를 벡터 표현(임베딩)으로 변환

### Decoder-only 모델
- **목적**: 주로 텍스트 생성 작업
- **예시**: GPT 계열, LLaMA
- **특징**: 자기회귀적(autoregressive) 방식으로 다음 토큰을 예측
- **작동 방식**: 이전 토큰들만 보고 다음 토큰을 예측 (단방향)

### Encoder-Decoder 모델
- **목적**: 번역, 요약 등 입력을 이해하고 출력을 생성하는 작업
- **예시**: T5, BART, 전통적인 번역 모델
- **작동 방식**: Encoder가 입력을 벡터로 변환하고, Decoder가 이 벡터를 기반으로 출력 생성

## Decoder-only 모델의 인코딩


귀하의 핵심 질문인 "decoder-only 모델에서는 인코딩 과정이 없는가?"에 대한 답은:

**인코딩 과정은 존재합니다.** 단, 여기서 인코딩은 두 가지 의미가 있습니다:

1. **토큰화 및 임베딩**: 모든 NLP 모델은 텍스트를 숫자 형태로 변환하는 과정을 거칩니다. 단어나 부분 단어를 토큰화하고 이를 벡터로 변환합니다.

2. **아키텍처로서의 인코더**: 이는 트랜스포머 구조에서 특정 역할을 하는 컴포넌트를 의미합니다.

Decoder-only 모델에서는 1번 의미의 인코딩(토큰화 및 임베딩)은 항상 존재하지만, 2번 의미의 독립적인 인코더 컴포넌트는 없습니다.

---
# N2N problem
## NER(Named Entity Recognition)
개체 중에서 필요한 정보만 뜯어내는 방식

# Morphology Analysis(형태소 분석기)
사실 n to m이라던가로 변할수도있음

# N2M problem
## Machine Translation 
## Dialog Mode
## Summarization
## Image Captioning

# NLP Benchmarks
크게 두가지 있음.
1 이게 상용화될 수 있을까?
2 이 모델이 언어를 이해하긴 하냐?
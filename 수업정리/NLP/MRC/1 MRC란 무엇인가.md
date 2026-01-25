![[[MRC] (1강) MRC Intro.pdf]]

# MRC(Machine Reading Comprehension)

주어진 지문(Context)을 이해하고, 주어진 질의(Query/Question)에 대한 답변을 추론하는 자연어처리 문제임. 컴퓨터가 인간과 같이 텍스트를 이해하고 정보를 추출하여 질문에 답할 수 있는 능력을 평가하는 중요한 AI 과제임.

## 종류

### Extractive Answer Datasets

질의에 대한 답이 항상 주어진 지문의 segment로 존재하는 경우임. 시스템은 정답이 포함된 지문의 특정 부분(span)을 정확히 찾아내야 함.

**예시 데이터셋**:

- SQuAD (Stanford Question Answering Dataset): 위키피디아 문서에서 추출한 질문-답변 쌍으로 구성됨
- KorQuAD: 한국어 기계 독해 데이터셋으로, SQuAD와 유사한 형식을 가짐
- NewsQA: 뉴스 기사 기반의 질문-답변 데이터셋
- Natural Questions: 구글 검색 쿼리와 위키피디아 문서로 구성된 대규모 데이터셋

### Descriptive/Narrative Answer Datasets

답이 지문 내에서 직접 추출한 것이 아니라, 지문을 이해한 후 질문에 대해 시스템이 자체적으로 답변을 생성하는 형태임. 주어진 정보를 종합하고 추론하여 자연어로 응답을 생성해야 함.

**예시 데이터셋**:

- MS MARCO: 마이크로소프트의 실제 검색 쿼리에 기반한 데이터셋
- NarrativeQA: 책과 영화 스크립트에 대한 포괄적인 이해가 필요한 질문
- CoQA (Conversational Question Answering): 대화 형식의 질의응답 데이터셋

### Multiple-choice Datasets

질문에 대한 답을 여러 개의 answer candidates 중 하나로 선택하는 형태임. 각 선택지를 평가하여 가장 적합한 답을 고르는 능력을 측정함.

**예시 데이터셋**:

- MCTest: 어린이 동화 기반의 독해력 테스트
- RACE: 중국 영어 시험에서 추출한 독해 문제
- SWAG: 상식 추론이 필요한 선택형 질문
- ARC (AI2 Reasoning Challenge): 과학 관련 문제로 구성된 난이도 높은 데이터셋

## Challenges in MRC

### 어휘 및 구문적 변형 이해

단어의 구성이 달라도 의미가 같은 표현을 이해해야 함. 동의어, 패러프레이징, 다양한 표현 방식을 인식하고 의미적 동등성을 파악하는 능력이 필요함.

**예시**:

- 질문: "아인슈타인은 언제 태어났는가?"
- 지문: "알베르트 아인슈타인의 출생일은 1879년 3월 14일이다."

여기서 "태어났다"와 "출생일"은 다른 표현이지만 같은 의미를 가짐.

### Unanswerable Questions

지문에 답이 없는(대답할 수 없는) 질문에 대해 "답변 불가"를 인식하는 능력임. 시스템은 답을 억지로 찾으려 하지 않고, 주어진 정보만으로는 답할 수 없음을 인지해야 함.

**예시 데이터셋**:

- SQuAD 2.0: 원래 SQuAD에 답변할 수 없는 질문들을 추가함
- DuoRC: 동일한 영화에 대한 두 개의 다른 줄거리를 사용하여 만든 데이터셋

### Multi-Hop Reasoning

여러 지문에 정보가 분산되어 있어, 여러 단계의 추론을 통해 답을 도출해야 하는 문제임. 여러 문장이나 단락에서 정보를 수집하고 연결하여 복잡한 질문에 답해야 함.

**예시 데이터셋**:

- HotpotQA: 위키피디아 문서에서 여러 단계 추론이 필요한 질문
- ComplexWebQuestions: 웹에서 복잡한 질문에 대한 답을 찾는 데이터셋
- IIRC (I'll Have What She's Having: Interactive Information Retrieval): 대화 맥락에서 여러 문서를 참조해야 하는 데이터셋

### 세계 지식과 상식 추론

텍스트에 명시적으로 언급되지 않은 배경 지식이나 상식을 활용해야 하는 문제임. 모델은 지문 외의 일반적인 지식을 갖추고 있어야 함.

**예시**:

- 질문: "사과가 떨어진 이유는 무엇인가?"
- 지문: "뉴턴은 나무 아래 앉아있을 때 사과가 떨어지는 것을 보았다."

여기서 "중력" 때문이라는 답을 추론하려면 물리학적 상식이 필요함.

### 시간적, 인과적 추론

사건의 시간적 순서나 인과 관계를 이해하고 추론해야 하는 문제임. 텍스트에 나타난 사건들 사이의 관계를 파악해야 함.

**예시 데이터셋**:

- TimeQA: 시간적 관계에 초점을 맞춘 질문-답변 데이터셋
- TORQUE: 시간적 순서에 대한 이해를 테스트하는 데이터셋

## MRC 평가방법

### Exact Match (EM)

시스템의 답변이 정답과 정확히 일치하는지 측정하는 방법임. 단 하나의 문자라도 다르면 0점, 정확히 일치하면 1점을 부여함.

**수식**: $EM = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(prediction_i = answer_i)$

여기서 $\mathbf{1}$은 지시함수로, 예측값과 정답이 정확히 일치할 경우 1, 그렇지 않으면 0을 반환함.

### F1 Score

Extractive MRC에서 많이 사용되는 평가 지표로, 예측 답변과 정답 간의 단어 중복을 측정함. 정밀도(Precision)와 재현율(Recall)의 조화 평균으로 계산됨.

**수식**: $Precision = \frac{|Prediction \cap Answer|}{|Prediction|}$ $Recall = \frac{|Prediction \cap Answer|}{|Answer|}$ $F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)

주로 Descriptive answer datasets에서 사용되는 평가 지표임. 예측 답변과 정답 간의 가장 긴 공통 부분 문자열(LCS)을 기반으로 유사도를 측정함.

**수식**: $R_{ROUGE-L} = \frac{|LCS(prediction, answer)|}{|answer|}$ $P_{ROUGE-L} = \frac{|LCS(prediction, answer)|}{|prediction|}$ $F_{ROUGE-L} = \frac{(1+\beta^2) \times R_{ROUGE-L} \times P_{ROUGE-L}}{R_{ROUGE-L} + \beta^2 \times P_{ROUGE-L}}$

여기서 $\beta$는 일반적으로 재현율에 더 가중치를 두기 위해 1.2로 설정함.

### BLEU (Bilingual Evaluation Understudy)

기계 번역에서 사용되는 평가 지표로, Descriptive answer datasets에서도 활용됨. n-gram 기반 정밀도를 측정하여 생성된 텍스트의 품질을 평가함.

**수식**: $BLEU = BP \times \exp(\sum_{n=1}^{N} w_n \log p_n)$

여기서 $BP$는 짧은 답변에 패널티를 주는 brevity penalty, $p_n$은 n-gram 정밀도, $w_n$은 각 n-gram에 대한 가중치임.

### Mean Reciprocal Rank (MRR)

Multiple-choice 문제에서 시스템이 정답을 얼마나 높은 순위에 배치했는지 평가하는 지표임.

**수식**: $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$

여기서 $|Q|$는 질문의 수, $rank_i$는 i번째 질문에 대한 정답의 순위임.

### Accuracy

Multiple-choice 데이터셋에서 가장 많이 사용되는 평가 지표로, 시스템이 정확히 맞춘 질문의 비율을 측정함.

**수식**: $Accuracy = \frac{Number\ of\ Correct\ Answers}{Total\ Number\ of\ Questions}$

## 최신 MRC 모델 구조와 기법

### Transformer 기반 모델

BERT, RoBERTa, ALBERT, T5 등 트랜스포머 아키텍처를 기반으로 한 사전 학습 언어 모델이 MRC 성능을 크게 향상시킴. 자기 주의(self-attention) 메커니즘을 통해 장거리 의존성을 효과적으로 포착할 수 있음.

### 지식 증강 기법

외부 지식 베이스나 검색 엔진을 통합하여 모델의 지식을 보강하는 방법임. Knowledge-Augmented Language Models은 특히 사실 기반 질문에 효과적임.

### 검색 증강 생성 (Retrieval-Augmented Generation, RAG)

대규모 외부 코퍼스에서 관련 문서를 검색하고, 이를 기반으로 답변을 생성하는 방법임. 지식 한계를 극복하고 최신 정보에 접근할 수 있게 해줌.

### 멀티모달 MRC

텍스트뿐만 아니라 이미지, 그래프, 표 등 다양한 형태의 정보를 함께 이해하는 모델임. 실제 세계의 복잡한 정보 이해에 필요한 방향임.

# Unicode / Tokenization

## Unicode?

유니코드(Unicode)는 전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 설계된 국제 표준 문자 인코딩 체계임. 각 문자마다 고유한 코드 포인트(code point)를 할당하여, 언어, 문자, 이모지 등을 통일된 방식으로 표현함.

유니코드 이전에는 ASCII(American Standard Code for Information Interchange)가 널리 사용되었으나, 영어 알파벳과 일부 특수 문자만 128개(또는 확장 ASCII의 경우 256개)로 제한되어 있어 다국어 처리에 한계가 있었음. 유니코드는 이러한 제한을 극복하고 백만 개 이상의 문자를 포함할 수 있도록 설계되었음.

### Encoding & UTF-8

유니코드 자체는 문자와 숫자(코드 포인트) 간의 매핑일 뿐, 실제로 컴퓨터가 이를 저장하려면 인코딩 방식이 필요함. 대표적인 유니코드 인코딩 방식으로는 UTF-8, UTF-16, UTF-32가 있음.

**UTF-8 (Unicode Transformation Format - 8-bit)**:

- 가변 길이 인코딩 방식으로, 문자에 따라 1~4바이트를 사용함
- ASCII 문자는 1바이트로 표현되어 기존 ASCII와 호환됨
- 영어 등 라틴 문자는 1바이트, 한글이나 중국어 등은 3바이트로 표현됨
- 웹에서 가장 널리 사용되는 인코딩 방식임
- 바이트 순서에 영향을 받지 않음(Byte Order Mark 불필요)

**UTF-8 인코딩 구조**:

- 1바이트: 0xxxxxxx (ASCII 호환)
- 2바이트: 110xxxxx 10xxxxxx
- 3바이트: 1110xxxx 10xxxxxx 10xxxxxx
- 4바이트: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

**UTF-16**:

- 대부분의 문자를 2바이트로 표현하며, 일부 특수 문자는 4바이트 사용
- 자바, 윈도우 내부 시스템에서 많이 사용됨
- 바이트 순서(Byte Order)가 중요하여 BOM(Byte Order Mark)이 필요함

**UTF-32**:

- 모든 문자를 고정 4바이트로 표현함
- 메모리 사용량이 많지만 처리 속도가 빠름
- 문자 인덱싱이 간단함

### Python에서 Unicode 다루기

파이썬 3.x는 기본적으로 문자열을 유니코드로 처리함. 내부적으로 모든 문자열은 유니코드 객체임.

**주요 함수와 메서드**:

```python
# 문자열을 바이트로 인코딩
text = "안녕하세요"
encoded = text.encode('utf-8')  # b'\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94'

# 바이트를 문자열로 디코딩
decoded = encoded.decode('utf-8')  # '안녕하세요'

# 유니코드 코드 포인트 확인
ord('가')  # 44032
chr(44032)  # '가'

# 유니코드 이스케이프 시퀀스
unicode_str = '\u00E9'  # 'é'
unicode_str = '\U0001F604'  # '😄'
```

**유니코드 정규화**: 문자는 여러 방식으로 표현될 수 있음(예: é는 단일 코드 포인트 또는 e와 ´의 조합으로 표현 가능). 정규화를 통해 일관된 표현을 보장함.

```python
import unicodedata

# NFD: 정준 분해 (문자를 기본 문자와 결합 문자로 분해)
nfd = unicodedata.normalize('NFD', 'é')  # 'e´'

# NFC: 정준 결합 (분해된 문자를 다시 결합)
nfc = unicodedata.normalize('NFC', 'e´')  # 'é'
```

### Unicode와 한국어

한글은 유니코드에서 다음과 같이 표현됨:

1. **완성형 한글 (Hangul Syllables)**:
    
    - 유니코드 범위: U+AC00 ~ U+D7A3
    - 11,172개의 모든 가능한 한글 음절을 완성형으로 표현
    - 예: '가'(U+AC00), '힣'(U+D7A3)
2. **조합형 한글 (Hangul Jamo)**:
    
    - 유니코드 범위: U+1100 ~ U+11FF (자모)
    - 초성, 중성, 종성을 개별적으로 표현
    - NFD 정규화 시 완성형 한글이 조합형으로 분해됨

한국어 처리 시 주의사항:

- 정규화: 같은 한글이라도 NFD와 NFC에 따라 다르게 표현될 수 있음
- 자모 분리: 형태소 분석 등에서 자모 단위 분석 시 유니코드 특성 고려 필요
- 인코딩: 한글은 UTF-8에서 3바이트를 차지하므로 메모리 사용량 고려 필요

## Tokenizing

토크나이징(Tokenizing)은 텍스트를 더 작은 단위(토큰)로 분리하는 과정임. 자연어처리(NLP)에서 텍스트 데이터를 모델이 처리할 수 있는 형태로 변환하는 첫 번째 단계임.

**토큰화의 종류**:

1. **단어 토큰화(Word Tokenization)**: 텍스트를 단어 단위로 분리
    
    ```
    "I love natural language processing." → ["I", "love", "natural", "language", "processing", "."]
    ```
    
2. **문장 토큰화(Sentence Tokenization)**: 텍스트를 문장 단위로 분리
    
    ```
    "Hello world. How are you?" → ["Hello world.", "How are you?"]
    ```
    
3. **문자 토큰화(Character Tokenization)**: 텍스트를 문자 단위로 분리
    
    ```
    "Hello" → ["H", "e", "l", "l", "o"]
    ```
    

**언어별 토큰화 특성**:

- **영어**: 공백과 구두점을 기준으로 비교적 쉽게 토큰화 가능
- **한국어**: 교착어 특성으로 형태소 분석이 필요한 경우가 많음
- **중국어/일본어**: 단어 간 공백이 없어 단어 경계 탐지가 필요함

**토큰화 도구**:

- NLTK, SpaCy, KoNLPy(한국어), MeCab(일본어) 등

### Subword Tokenizing

서브워드 토크나이징은 단어보다 작은 의미 있는 단위로 텍스트를 분할하는 방법임. 이는 다음과 같은 장점을 제공함:

1. **어휘 크기 감소**: 모든 단어를 개별 토큰으로 처리하는 대신, 공통 부분을 재사용
2. **미등록 단어(OOV) 문제 해결**: 새로운 단어도 서브워드 단위로 분해하여 처리 가능
3. **형태학적 정보 포착**: 접두사, 접미사 등의 형태소 정보를 암묵적으로 학습

**대표적인 서브워드 토크나이징 방법**:

- BPE (Byte-Pair Encoding)
- WordPiece (BERT에서 사용)
- Unigram Language Model (SentencePiece에서 구현)
- SentencePiece (Google의 토크나이저)

### BPE(Byte-Pair Encoding)

BPE는 가장 널리 사용되는 서브워드 토크나이징 알고리즘 중 하나임. 원래 데이터 압축을 위해 개발되었으나, NLP에 효과적으로 적용됨.

**BPE 알고리즘 과정**:

1. 모든 단어를 문자(또는 바이트) 시퀀스로 분리하고, 각 문자에 빈도수 부여
2. 가장 빈번하게 함께 등장하는 문자 쌍(byte pair)을 찾음
3. 이 쌍을 새로운 토큰으로 병합
4. 2-3단계를 원하는 어휘 크기에 도달할 때까지 반복

**BPE 예시**:

```
# 초기 단어 리스트
["low", "lower", "newest", "widest"]

# 문자 단위로 분할 (마지막 문자에 </w> 추가하여 단어 끝 표시)
l o w </w>
l o w e r </w>
n e w e s t </w>
w i d e s t </w>

# 빈도수 계산 후 가장 많이 등장하는 쌍 병합 (예: e, s)
→ 'es' 토큰 생성

# 계속 반복...
```

**BPE의 장점**:

- 데이터에 적응적: 훈련 데이터에 기반하여 최적의 서브워드 단위 학습
- 희귀 단어/합성어 처리에 효과적: "unsupervisedness" → "un" + "super" + "vised" + "ness"
- 다국어 처리에 적합: 언어별 특성을 고려한 토큰화 가능

**실제 활용**:

- GPT 모델 시리즈는 BPE를 기반으로 한 토크나이저 사용
- 최신 모델들은 종종 문자 수준이 아닌 바이트 수준에서 BPE 적용 (Byte-level BPE)

**한국어 BPE 적용 특성**:

- 한글의 교착어 특성으로 인해 자주 사용되는 조사, 어미 등이 별도 서브워드로 분리됨
- 음절 단위로 적용 시 한글의 형태적 특성을 효과적으로 포착할 수 있음
- 자모 단위로 적용 시 극도로 희귀한 한글 음절도 처리 가능하나 토큰 시퀀스가 길어짐

토크나이징 성능은 자연어처리 태스크 성능에 직접적인 영향을 미치므로, 언어와 태스크 특성에 맞는 토크나이징 방법 선택이 중요함.


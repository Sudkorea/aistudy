![[[MRC] (2강) Extraction-based MRC.pdf]]

# Extraction-based MRC
## 정의
질문에 대한 답변이 항상 주어진 지문 내에 존재함.

## 평가 방법

### EM score
exact match : 정확히 같아야 1점 아니면 0점
### F1 Score
예측값과 정답의 overlap 비율로 계산 함


# Pre-Processing
## Tokenization
텍스트를 작은 단위로 나누는거

## Special Tokens

## Attention Mask
입력 시퀀스 중 attention 연산할 때 무시할거 표시해놓음. padding 토큰 무시하는 등

## Token Type IDs
입력이 2개 이상 시퀀스일 때 각각 ID값 주는거

## 모델 출력값
정답은 문서 안에 존재하는 연속된 단어 토큰(span)이므로, span의 시작, 끝만 알면 됨.
Extraction-Based에선 답안 생성보단, 시작과 끝 토큰 위치 정해주는걸 학습함. 즉, Token-Classification문제로 치환됨.

# Fine-Tuning

## BERT

# Post-processing

## 불가능한 답 제거
end position이 start position보다 앞에 이쓴 경우
예측한 위치가 context 벗어난 경우
미리 설정한 max length 벗어난 경우

## 최적의 답안 찾기


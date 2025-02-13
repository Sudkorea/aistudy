![[[NLP 이론] (1강) Tokenization.pdf]]

## Tokenization
token : 자연어 처리 모델이 각 타임스텝에서 주어지는 각 단어로 다루는 단위

Tokenization은 크게 3가지 방식으로 구분함

### 1. Word-level Tokenization
단어 단위로 토큰을 구분하는거임
영어는 띄어쓰기 기준으로 구분하는데, 한국어의 경우 형태소 기준으로 나누기도 함(이건 언어마다 다름)

문제점 : 사전에 없는 단어가 등장하는 경우, 모두 Unknown으로 처리하고 넘어감

### 2. Character
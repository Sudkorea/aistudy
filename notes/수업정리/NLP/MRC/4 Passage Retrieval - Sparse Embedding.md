![[[MRC] (4강) Passage Retrieval-Sparse Embedding.pdf]]

# Introduction to Passage Retrieval
## 정의
질문(query)에 맞는 문서(passage)를 찾는 것

## with MRC
Open-domain Question Answering : 대규모의 문서 중에서 질문에 대한 답 찾기
둘이 섞어서 2-stage로 돌아가면?
Query - Passage embedding 이후 유사도로 랭킹 매기고, 유사도 가장 높은 passage 선택

# Passage Embedding and Sparse Embedding
Passage Embedding의 벡터공간
벡터화되었으니 유사도계산 같은거 가능

## BoW 구성하는 법 : n-gram
unigram(n=1)
bigram(n=2)
## Term Value를 결정하는 방법
Term이 document에 등장하는가
Term이 몇 번 등장하는가

## 특징
1. Dim = number of terms
   등장하는 단어가 많을수록
   n이 커질수록
2. Term overlap 정확하게 잡아내는 장점이 있음
3. 의미가 비슷하지만 다른 경우일 때 비교가 안됨
# TF-IDF(Time Frequency - Inverse Document Frequency)
Term 셀때 존재하나 안하나가 vanila 방법이라 했잖음
- TF : 단어의 등장 빈도
- IDF : 단어가 제공하는 정보의 양

## Time Frequency
log normalization 등 방법론은 많음

## Inverse Document Frequency
$$IDF(t)=\log{\frac{N}{DF(t)}}$$
$DF$ = Term t가 등장한  document의 개수
$N$ = 총 Document의 개수

## 이 둘을 섞자
TF-IDF(t, d): TF-IDF for term t in document d
$$TF(t, d)\times IDF(t)$$






![[[NLP 이론] (9강) Self-Supervised Pre-trained model - BERT.pdf]]

- Self-supervised learning과 Transfer learning의 개념 
	모자이크 복원, 이미지 일부 조각내고 예측하도록 학습하기, 
	Pre-training
	Transfer learning
- BERT의 사전학습 방법론과 구조적 특징 
	Bidirectional Encoder Repersentations from Transformers
	대형 Transformer encoder
	MLM(Masked Language Model)
		무작위로 입력 토큰 중 15%정도 감추고 예측함.
		15% 중 80%는 MASK시켜서 지우고
		10%는 무작위 토큰으로 바꾸고
		10%는 원래 토큰 사용함.(이거도 무작위로 바꿨다가 편향 생길수도 있음)
	input embedding = Token + Segment + Position Encoding

- BERT를 이용한 다양한 자연어 처리 문제 수행 방법
	Downstream Task : Sentence Classification
	주어진 한 문장에 대해 기준에 맞는 분류를 수행하는 Task
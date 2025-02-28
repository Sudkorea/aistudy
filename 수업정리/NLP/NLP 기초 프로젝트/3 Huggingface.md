![[[NLP 기초프로젝트] (3강) Huggingface 소개.pdf]]

트렌드가 점점 사람의 직관이 들어갈 부분이 없어지고있음
모든게 결국 모델 하나로 통일되어가는 감이 있음

아키텍쳐끼리 경쟁하던 야생의 시기가 있었지만, 트랜스포머 이후로 모든게 하나로 수렴됨.

그래서 HF또한 친해져야함 트랜스포머 친해져야 하는 것과 같음.

# Hugging Face?

pytorch <-> tensorflow 양방향 바꿔주는 툴로 시작했지만, 지금은 다 커버하는 툴로 진화함

## Data

### datasets
이거저거 데이터 다 클라우드에 관리하고, 다 써보라고 줌. 미친빛빛빛GODGODGODGOD
데이터 불러와서
tokenizer 하고
dataset mapping 해서
datacollator까지
이걸 다해줌
### Tokenizer
모델마다 맞는 토큰이 있음. 검색해보고 쓸것
padding 쓸건지 말건지 등등등
### dataCollator


## Model

### Tansformers



### PEFT(Parameter Efficient Fine Tuning)
개인 환경에서 돌려보려면 최적화 해야하니까, 그거 도와주는 툴임
### Bitsandbytes(META)

## Training

### Trainer
train할때 이것저것 다 건드려볼 수 있음
### accerlate
분산학습
## Evaluate

### evaluate

## Share

### pipeline


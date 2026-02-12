
# SALMONN 기본 구조 분석

## 1. `models/__init__.py`
```python
from .salmonn import SALMONN

def load_model(config):
    return SALMONN.from_config(config)
```

### 주요 기능
- 모델 로딩의 진입점 역할
- `from_config` 메소드를 통해 설정 기반 모델 초기화
- 단순한 인터페이스로 모델 생성 과정 추상화

## 2. config.py

### Config 클래스 구조
```python
class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        config = OmegaConf.merge(config, user_config)
        self.config = config
```

### 주요 메소드
1. `_convert_to_dot_list`
```python
def _convert_to_dot_list(self, opts):
    # 커맨드라인 옵션을 OmegaConf 형식으로 변환
    if opts is None:
        return []
    has_equal = opts[0].find("=") != -1
    if has_equal:
        return opts
    return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]
```

2. `_build_opt_list`
```python
def _build_opt_list(self, opts):
    # 옵션을 dotlist 형식으로 변환하고 OmegaConf 객체 생성
    opts_dot_list = self._convert_to_dot_list(opts)
    return OmegaConf.from_dotlist(opts_dot_list)
```

3. `pretty_print`
```python
def pretty_print(self):
    # 설정 정보를 보기 좋게 출력
    logging.info("\n=====  Running Parameters    =====")
    logging.info(self._convert_node_to_json(self.config.run))
    logging.info("\n======  Dataset Attributes  ======")
    logging.info(self._convert_node_to_json(self.config.datasets))
    logging.info("\n======  Model Attributes  ======")
    logging.info(self._convert_node_to_json(self.config.model))
```

### 설정 구조
1. 실행 파라미터 (run)
   - 학습 관련 설정
   - 분산 학습 설정
   - 디바이스 설정

2. 데이터셋 설정 (datasets)
   - 데이터 경로
   - 데이터 처리 파라미터

3. 모델 설정 (model)
   - 모델 아키텍처
   - 하이퍼파라미터
   - 사전 학습 모델 경로

### 작동 방식
1. 설정 초기화
```python
# 커맨드라인 인자로 받은 설정 파일 로드
config = OmegaConf.load(self.args.cfg_path)

# 사용자 정의 옵션과 병합
config = OmegaConf.merge(config, user_config)
```

2. 설정 접근
```python
# 계층적 구조로 설정값 접근 가능
run_config = cfg.config.run
model_config = cfg.config.model
data_config = cfg.config.datasets
```

3. JSON 변환
```python
def _convert_node_to_json(self, node):
    container = OmegaConf.to_container(node, resolve=True)
    return json.dumps(container, indent=4, sort_keys=True)
```

### 특징
- OmegaConf 사용으로 유연한 설정 관리
- 커맨드라인 옵션 오버라이드 지원
- 계층적 구조로 설정 관리
- 깔끔한 로깅 지원

이러한 구조를 통해 모델의 다양한 설정을 효율적으로 관리하고, 실험을 쉽게 수행할 수 있도록 지원합니다.



# SALMONN Dataset 분석

## SALMONNDataset 클래스 구조

### 초기화
```python
class SALMONNDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__()
        self.prefix = prefix  # 데이터 경로 prefix
        self.annotation = json.load(open(ann_path, "r"))["annotation"]  # 어노테이션 로드
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)  # Whisper 특징 추출기
```

### 주요 메소드

1. `__len__`
```python
def __len__(self):
    return len(self.annotation)  # 데이터셋 크기 반환
```

2. `__getitem__`
```python
def __getitem__(self, index):
    ann = self.annotation[index]
    audio_path = self.prefix + '/' + ann["path"]
    
    # 오디오 로드 및 전처리
    audio, sr = sf.read(audio_path)
    
    # 스테레오를 모노로 변환
    if len(audio.shape) == 2:
        audio = audio[:, 0]
        
    # 최소 1초 길이 보장을 위한 패딩
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
        
    # 샘플링 레이트 조정
    if sr != self.wav_processor.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)
        sr = self.wav_processor.sampling_rate
        
    # 30초로 오디오 길이 제한
    audio = audio[: sr * 30]
    
    # 스펙트로그램 생성
    spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
    
    return {
        "spectrogram": spectrogram,
        "raw_wav": audio,
        "text": ann["text"],
        "task": ann.get("task", "asr"),  # 기본값 "asr"
        "Q": ann.get("Q", ""),  # 기본값 빈 문자열
        "id": ann["path"]
    }
```

3. `collater` (배치 처리)
```python
def collater(self, samples):
    # 스펙트로그램 배치 처리
    samples_spectrogram = [s["spectrogram"] for s in samples]
    cat_spectrogram = torch.stack(samples_spectrogram, dim=0)
    
    # raw 웨이브폼 배치 처리
    raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
    raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
    raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
    
    # 패딩 마스크 생성
    paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
    
    return {
        "spectrogram": cat_spectrogram,
        "raw_wav": raw_wav,
        "padding_mask": paddding_mask,
        "text": [s["text"] for s in samples],
        "task": [s["task"] for s in samples],
        "Q": [s["Q"] for s in samples],
        "id": [s["id"] for s in samples]
    }
```

### 데이터 처리 파이프라인

1. 오디오 전처리
   - 스테레오 → 모노 변환
   - 최소 길이(1초) 보장을 위한 패딩
   - 샘플링 레이트 조정
   - 최대 길이(30초) 제한

2. 특징 추출
   - Whisper 특징 추출기를 사용한 스펙트로그램 생성
   - raw 웨이브폼 보존

3. 배치 처리
   - 스펙트로그램 스택킹
   - 웨이브폼 패딩
   - 패딩 마스크 생성
   - 텍스트 및 메타데이터 수집

### 데이터 구조

1. 입력 어노테이션 형식
```json
{
    "annotation": [
        {
            "path": "audio_file_path",
            "text": "transcription_or_description",
            "task": "task_type",  // optional
            "Q": "question"       // optional
        }
        // ...
    ]
}
```

2. 반환 데이터 형식
```python
{
    "spectrogram": torch.Tensor,      # [batch_size, time, feature_dim]
    "raw_wav": torch.Tensor,          # [batch_size, max_length]
    "padding_mask": torch.Tensor,     # [batch_size, max_length]
    "text": List[str],               # [batch_size]
    "task": List[str],               # [batch_size]
    "Q": List[str],                  # [batch_size]
    "id": List[str]                  # [batch_size]
}
```

### 특징
- Whisper 특징 추출기 통합
- 유연한 오디오 전처리
- 효율적인 배치 처리
- 다양한 태스크 지원 (ASR, QA 등)
- 안정적인 예외 처리



SALMONN 모델의 아키텍처를 분석해보겠습니다. 핵심 파일인 `models/salmonn.py`를 중심으로 설명하겠습니다.

# SALMONN 모델 아키텍처

## 1. 전체 구조
````python
class SALMONN(nn.Module):
    def __init__(
        self,
        llama_path="",          # LLaMA 모델 경로
        whisper_path="",        # Whisper 모델 경로
        beats_path="",          # BEATs 모델 경로
        use_speech_Qformer=True,# Qformer 사용 여부
        lora=True,             # LoRA 적용 여부
        multi_prompt=False,     # 다중 프롬프트 사용 여부
        ...
    ):
````

### 주요 컴포넌트:
1. **인코더**
   - Whisper: 음성 인식
   - BEATs: 오디오 이해
   
2. **연결 계층**
   - Qformer: 모달리티 간 연결
   - Speech-LLaMA Projection: LLaMA 임베딩 공간으로 투영

3. **디코더**
   - LLaMA: 텍스트 생성

## 2. 데이터 흐름

### 음성 처리 파이프라인:
````python
# 1. Whisper 인코딩
speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
speech_embeds = self.speech_encoder(spectrogram)
speech_embeds = self.ln_speech(speech_embeds)

# 2. BEATs 인코딩 (선택적)
if self.beats_path:
    audio_embeds = self.beats(raw_wav, padding_mask)
    audio_embeds = self.ln_audio(audio_embeds)
````

### 모달리티 연결:
````python
# Qformer를 통한 특징 융합
if self.use_speech_Qformer:
    # 쿼리 토큰 초기화
    query_tokens = self.speech_query_tokens.expand(B, -1, -1)
    
    # Whisper와 BEATs 특징 결합
    if audio_embeds is not None:
        speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
        
    # Qformer 처리
    speech_embeds = self.speech_Qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=speech_embeds,
        encoder_attention_mask=speech_atts,
    )[0]
    
    # LLaMA 임베딩 공간으로 투영
    speech_embeds = self.speech_llama_proj(speech_embeds)
````

### 텍스트 생성:
````python
# LLaMA 모델을 통한 텍스트 생성
outputs = self.llama_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_length=max_length,
    num_beams=num_beams,
    min_length=min_length,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    length_penalty=length_penalty,
    temperature=temperature,
)
````

## 3. 주요 특징

### LoRA 적용:
````python
if self.lora:
    self.peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=lora_rank,            # LoRA 랭크
        lora_alpha=lora_alpha,  # LoRA 스케일링
        lora_dropout=lora_dropout,
    )
    self.llama_model = get_peft_model(self.llama_model, self.peft_config)
````

### 윈도우 레벨 Qformer:
````python
if self.window_level_Qformer:
    # 오디오를 윈도우 단위로 처리
    kernel = round(1500 * self.second_per_window / 30.0)
    stride = round(1500 * self.second_stride / 30.0)
    speech_embeds_overlap = F.unfold(
        speech_embeds_tr, 
        kernel_size=kernel, 
        stride=stride
    )
````

### 프롬프트 처리:
````python
def _get_prompt(self, task=None):
    if self.multi_prompt:
        # 태스크별 프롬프트 선택
        prompts = self.prompt_dict.get(task, self.prompt_dict["default"])
        return random.choice(prompts)
    return None
````

## 4. 아키텍처 특징

1. **모듈성**
   - 각 컴포넌트가 독립적으로 구성
   - 쉬운 확장과 수정 가능

2. **효율성**
   - LoRA를 통한 효율적인 파인튜닝
   - 윈도우 기반 처리로 긴 시퀀스 처리 가능

3. **유연성**
   - 다양한 태스크 지원
   - 설정을 통한 쉬운 모델 구성

4. **확장성**
   - 새로운 모달리티 추가 용이
   - 다양한 프롬프트 전략 지원

이러한 아키텍처를 통해 SALMONN은 음성과 텍스트를 효과적으로 연결하여 다양한 멀티모달 태스크를 수행할 수 있습니다. 특히 Qformer를 통한 모달리티 연결과 LoRA를 통한 효율적인 학습이 주요 특징입니다.



# SALMONN 학습 파이프라인 분석

## 1. train.py (학습 진입점)

### 주요 구성
````python
def main():
    # 1. 초기화
    job_id = now()  # 작업 ID 생성
    
    # 2. 설정 로드
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets
    
    # 3. 분산 학습 설정
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger()
    
    # 4. Wandb 로깅 설정
    if global_rank == 0:
        wandb.login()
        wandb.init(project="audio_lm", name=run_config.exp_name)
    
    # 5. 데이터셋 구성
    datasets = {
        "train": SALMONNDataset(...),
        "valid": SALMONNDataset(...),
        "test": SALMONNDataset(...)
    }
    
    # 6. 모델 생성
    model = load_model(model_config)
    
    # 7. Runner 생성 및 학습 시작
    runner = Runner(cfg, model, datasets, job_id, args.dryrun)
    runner.train()
````

### 주요 기능
1. **설정 관리**
   - 커맨드라인 인자 파싱
   - 설정 파일 로드
   - 하이퍼파라미터 관리

2. **분산 학습 설정**
   - 멀티 GPU 학습 지원
   - 시드 설정
   - 로깅 설정

3. **실험 관리**
   - Wandb 연동
   - 작업 ID 관리
   - 로깅 시스템

## 2. runner.py (학습 프로세스 관리)

### Runner 클래스 구조
````python
class Runner:
    def __init__(self, cfg, model, datasets, job_id, dryrun):
        self.config = cfg
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.log_writter = SummaryWriter(self.output_dir)
        
        # 학습 설정
        self.device = torch.device(self.config.config.run.device)
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.cuda_enabled = (self.device.type == "cuda")
````

### 주요 메소드

1. **학습 루프**
````python
def train(self):
    # 옵티마이저 및 스케줄러 설정
    self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
    self.scheduler = LinearWarmupCosineLRScheduler(
        self.optimizer,
        self.max_epoch,
        self.config.config.run.optims
    )
    
    # 메인 학습 루프
    for epoch in range(self.start_epoch, self.max_epoch):
        # 데이터로더 에포크 설정
        if self.use_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            
        # 학습 단계
        train_stats = self.train_epoch(epoch)
        
        # 검증 단계
        if self.valid_loader:
            val_stats = self.valid_epoch(epoch)
            
        # 체크포인트 저장
        self.save_checkpoint(epoch)
````

2. **학습 에포크**
````python
def train_epoch(self, epoch):
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Train Epoch: [{epoch}]'
    
    for i, samples in enumerate(metric_logger.log_every(self.train_loader, header=header)):
        # 데이터 전처리
        samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
        
        # Forward 패스
        with self.autocast():
            loss = self.model(samples)
            
        # Backward 패스
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 메트릭 기록
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
````

3. **검증 에포크**
````python
def valid_epoch(self, epoch):
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    
    for samples in metric_logger.log_every(self.valid_loader, header=header):
        samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
        
        with torch.no_grad():
            loss = self.model(samples)
            
        metric_logger.update(loss=loss.item())
````

### 주요 기능

1. **최적화 관리**
   - AdamW 옵티마이저
   - Warmup과 Cosine LR 스케줄링
   - Mixed Precision 학습

2. **분산 학습 지원**
   - DistributedDataParallel 래핑
   - 데이터 샘플러 관리
   - 동기화 관리

3. **체크포인트 관리**
````python
def save_checkpoint(self, cur_epoch, is_best=False):
    model_no_ddp = self.unwrap_dist_model(self.model)
    state_dict = {
        "model": model_no_ddp.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "config": self.config.to_dict(),
        "scaler": self.scaler.state_dict(),
        "epoch": cur_epoch
    }
    torch.save(state_dict, checkpoint_path)
````

4. **로깅 시스템**
   - TensorBoard 로깅
   - Wandb 연동
   - 메트릭 추적
   - 진행률 표시

### 특징
1. **효율성**
   - Mixed Precision 학습
   - 효율적인 메모리 관리
   - 분산 학습 최적화

2. **유연성**
   - 다양한 데이터셋 지원
   - 설정 기반 학습 관리
   - 모듈화된 구조

3. **안정성**
   - 체크포인트 저장/로드
   - 예외 처리
   - 진행 상황 모니터링

이러한 학습 파이프라인을 통해 SALMONN 모델은 효율적이고 안정적인 학습을 수행할 수 있습니다.


**개념  
**

- 모델 구조 자체를 경량화된 네트워크(MobileNet, EfficientNet, ShuffleNet 등)로 교체하거나, Transformer 계열이라면 다른 구조 개선 연구(예: Linformer, Performer 등)를 참고
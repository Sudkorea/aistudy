## **FT-Transformer: A Transformer Architecture for Tabular Data 논문 요약**  

---

### **1. FT-Transformer가 뭐임?**  
- FT-Transformer는 범주형(categorical)과 수치형(numerical) 데이터를 동시에 처리하는 Transformer 모델
- 테이블 데이터(예: 금융, 부동산)를 다루는 데 강점이 있으며, 기존의 트리 기반 모델(XGBoost, LightGBM)보다 더 복잡한 상호작용을 학습함.

---

### **2. 핵심 특징**  
1. 범주형 데이터 임베딩
   - 범주형 데이터를 Feature Tokenizer로 벡터화해서 입력으로 사용함.  
   - 이 과정에서 범주형 데이터의 의미를 보존하고, 고차원 공간에서 정보 손실을 줄임.  

2. Self-Attention을 통한 상호작용 학습
   - 변수 간의 상관관계를 Self-Attention 메커니즘을 이용해 학습함.  
   - 범주형과 수치형 데이터 사이의 복잡한 상호작용을 포착하는 데 유리함.  

3. 다양한 데이터 융합
   - FT-Transformer는 수치형 데이터와 범주형 데이터를 자연스럽게 결합해 처리함.  
   - 외부 데이터도 쉽게 통합할 수 있어 다양한 도메인에 적용 가능함.

---

### **3. FT-Transformer 구조**  
1. Feature Tokenizer Layer
   - 범주형 데이터를 벡터로 변환해서 Transformer의 입력으로 제공함.  
2. Transformer Encoder Layer
   - Self-Attention과 Feedforward Layer로 구성되며, 각 변수 간 관계를 학습함.  
3. Output Layer
   - 최종 예측 결과를 출력함.

---

### **4. FT-Transformer의 필요성**  
- 범주형-수치형 데이터의 조합을 효율적으로 학습할 수 있음.  
- 기존의 트리 모델과 달리 변수 간의 복잡한 상호작용까지 반영해 더 높은 성능을 보일 수 있음.  
- 확장성이 높아 외부 데이터(예: 금리, 입지 정보)를 활용해 성능을 극대화할 수 있음.

---

### **5. 실제 적용 예시**  
- 부동산 예측: 건물 면적, 층수, 계약 정보와 금리 데이터를 결합해 예치금(deposit)을 예측함.  
- 추천 시스템: 사용자와 상품의 다양한 특성을 학습해 개인 맞춤형 추천을 제공함.

---

### **6. 결론 및 요약**  
- FT-Transformer는 범주형과 수치형 데이터를 동시에 처리할 수 있는 강력한 모델임.  
- Self-Attention 메커니즘을 통해 데이터 간의 중요한 상호작용을 놓치지 않음.  
- 여러 외부 데이터를 통합해 활용할 수 있어 다양한 예측 모델에 적합함.


---
## model_training.py를 분석해보자

### 1. 모델 아키텍처 (FTTransformer 클래스)
```python:src/model_training.py
class FTTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, d_model, d_ff, dropout):
        super(FTTransformer, self).__init__()
        # 1. 입력 임베딩 레이어
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. 트랜스포머 인코더 레이어 설정
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,      # 모델의 차원
            nhead=num_heads,      # 멀티헤드 어텐션의 헤드 수
            dim_feedforward=d_ff, # 피드포워드 네트워크의 차원
            dropout=dropout,      # 드롭아웃 비율
            batch_first=True      # 배치 차원을 첫번째로 설정
        )
        
        # 3. 전체 트랜스포머 인코더 구성
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 출력 레이어
        self.fc = nn.Linear(d_model, 1)
```

### 2. 모델 학습 함수 (train_ft_transformer)
```python:src/model_training.py
def train_ft_transformer(X_train, y_train, X_holdout, y_holdout, config):
    # 1. 데이터 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    
    # 2. 데이터로더 설정
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 3. 모델 초기화
    model = FTTransformer(
        input_dim=X_train.shape[1],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        dropout=config['dropout_rate']
    ).to(device)
    
    # 4. 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

### 3. 평가 함수 (evaluate_model)
```python:src/model_training.py
def evaluate_model(model, X_holdout, y_holdout):
    # 1. 예측 수행
    predictions = []
    with torch.no_grad():
        for batch_X, batch_y in holdout_loader:
            batch_pred = model(batch_X).cpu().numpy()
            predictions.extend(batch_pred)
    
    # 2. 평가 지표 계산
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
```

### 주요 구성 요소 설명:

1. **모델 구조**:
- 입력 임베딩: 원본 특성을 d_model 차원으로 변환
- 트랜스포머 인코더: 멀티헤드 어텐션과 피드포워드 네트워크로 구성
- 출력 레이어: 최종 예측값 생성

2. **학습 프로세스**:
- 데이터 준비: 텐서 변환 및 데이터로더 설정
- 모델 초기화: 하이퍼파라미터 설정
- 학습 루프: 배치 단위 학습 및 검증
- 학습률 조정: ReduceLROnPlateau 스케줄러 사용

3. **평가 과정**:
- 홀드아웃 세트에 대한 예측
- MAE, RMSE, R² 지표 계산

이 모델은 특히 테이블 데이터에 대한 회귀 문제를 해결하도록 설계되었으며, 트랜스포머의 셀프어탠션 메커니즘을 활용하여 특성 간의 복잡한 관계를 학습함


## 실험

**MAE vs. MSE 학습의 차이**

- **MAE로 학습**하면 **모든 데이터 포인트에 대해 균등한 중요도를 부여**하여, 극단적인 오차에 덜 민감하게 반응함.
- **MSE로 학습**하면 **큰 오차에 대해 더 큰 패널티를 부여**하여 큰 오차를 줄이려는 학습이 이루어짐.

MAE로 학습하는 경우
```
MAE: 8232.29  

RMSE: 16035.79  

R²: 0.64
```
MSE로 학습하는 경우
```
MAE: 8026.16  

RMSE: 14510.16  

R²: 0.70
```

- 이번 경우 MSE로 학습했을 때 MAE도 더 낮아졌으므로, MSE로 학습하는 것이 더 나은 선택이 될 수 있음.
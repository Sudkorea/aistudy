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


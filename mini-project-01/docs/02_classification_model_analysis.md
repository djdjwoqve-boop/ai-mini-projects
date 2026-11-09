# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** 더미,로직스틱
- **DummyClassifier — Validation Accuracy / F1:** 0.845	 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.734/ 0.367	
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181	 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
값이 불균형 하기 때문 재현율이 낮기 때문에 F1값이 낮기 때문입니다.

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 21
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.693	0.628
- **MLP-A Validation Precision / Recall:** 0.723	0.556

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  

그래프를 보니 대략 에포크 10에서 대략적으로 학습이 대부분은 정해진 걸로 보입니다.
검증 데이터는 성능이 개선되지 않는 반면에 훈련 데이터는 약간이지만 성능이 개선 되는 현상이 있어서 약간 과적합 경향이 있는 걸로 보입니다.
에포크 21에서 최적 성능을 내고 종료합니다 

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:**	0.684 / 0.631
- **MLP-B Validation Precision / Recall:** 0.556	0.731
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.062,0.052
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** 재현율에서 큰 차가 발생했습니다 
- **Train·Validation F1 차이에서 관찰한 점:** MLP 모델이 더미모델들과 비교해서 성능이 좋습니다.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
재현율이 올라가고 정밀도가 떨어졌습니다 F1스코어의 경우 훈련에 A가 이기고 검증에 B가 더 우수했습니다.

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 61	127
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 167	77
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** 정밀도 하락,재현율 증가 FP 증가 FN 감소
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** 정밀도 상승,재현율 하락, FP감소 FN증가 
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B
- **선택한 MLP의 최적 에포크 수:** 0.7
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** 17

**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다.  
최종 MLP: MLP-B (weighted BCE)
최종 임계값: 0.7
재학습 에포크: 17
이때 B의 결과는 	MLP-B (weighted BCE)	0.7	0.647	0.647	0.647	101	101입니다.
다른 모델과 비교해서 F1-score가 가장 높고 정밀도,재현율이 동일하여 균형이 좋습니다. 
또한 FP,FN이 각각 101건으로 두 오류의 개수가 같다는 점이 이 모델의 선택을 결정한거 같습니다 

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** B / 0.7
- **최종 재학습 데이터(Train + Validation):** 10480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / yes
- **Test Accuracy / Precision / Recall / F1:** 0.886 / 0.610   /0.738  / 0.668
- **Test 혼동행렬:** TN 1429	 / FP135/ FN75	 / TP 211

- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 최종 MLP: MLP-B (weighted BCE) / threshold: 0.7
- **`predict_purchase()` 실행 결과:** True

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
잘 모르겠습니다. 

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** Reacll
- **남아 있는 모델의 한계:** 아직 전반적으로 정확도,재현율,정밀도가 낮습니다. 이러한 결과는 모델의 문제일수도 있지만 데이터에 대한 특징에 대한 이해가 부족해서 전처리가 부족해서 그렇지 않을까 싶습니다.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.

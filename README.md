<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163712073-7d2dcd09-4c1f-4bab-935f-42de292300bb.png" /></p>

<div align="center">
05 TEAM 알잘딱깔센 <br/>
  
## Deep Knowledge Tracing
  
</div>

# 🏆️ 프로젝트 목표
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168472960-0eac76e2-4fe3-4ebc-b093-f9c0aab59859.png" /></p>
- 유저의 문제 풀이 Sequence를 이용하여 유저의 지식 상태를 추론해, 유저가 마지막에 푼 문제를 틀릴지 맞출지를 예측

# 💻 활용 장비
- Ubuntu 18.04.5 LTS
- GPU Tesla V100-PCIE-32GB

# 🙋🏻‍♂️🙋🏻‍♀️ 프로젝트 팀 구성 및 역할
- **김건우**: EDA를 통한 feature 탐색, Project Template 탐색
- **김동우**: 베이스 모델 탐색
- **박기정:** EDA를 통한 Feature 탐색, Project Template 설계 및 제작, MLflow model registry 구축
- **심유정:** EDA를 통한 Feature 탐색, 베이스 모델 탐색, LGBM 모델 학습 및 Ensemble
- **이성범:** 데이터 패턴 분석, 모델 설계 및 학습, Ensemble, Project Template 제작

# ✏️ Model Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473170-938e1ce0-395f-40be-9118-ea127668b11d.png" /></p>

- 과거와 현재 풀이 정보의 연관성을 표현할 수 있는 Model Architecture를 설계
- 시간적 순서를 효과적으로 표현하기 위하여 Transformer와 LSTM을 활용
- 과거 풀이 정보와 현재 풀이 정보를 서로 다른 Embedding을 활용해 학습

# 🛠 Project Template
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473184-7a7a5c9b-f7da-4d92-81d8-965ecd1f934f.png" /></p>

- 학습 환경의 경우 [pytorch-template](https://github.com/victoresque/pytorch-template)을 이용하여 DKT 학습환경에 맞추어 리팩토링을 진행함

# 🎥 프로젝트 수행 결과 - private 3위
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473055-047f5162-a1f5-4c64-a5a5-275bb87aa744.png" /></p>

- 유저와 문항 정보를 함께 사용한 GMF 모델을 통해서 0.7429의 성능을 얻음
- 문항에 대한 정보만을 활용한 GMF 모델을 통해서 0.8258의 성능을 얻음
- 시간적 순서를 고려한 Transformer을 통해서 0.8302의 성능을 얻음
- Transformer와 LSTM을 함께 사용하여 0.8511의 성능을 얻음
- 과거 정보에 정답에 대한 Embedding을 추가하여 0.8636의 성능을 얻음
- 과거 정보와 현재 정보를 같이 Modellig 하여 0.8590의 성능을 얻음
- 과거 정보와 현재 정보를 서로 다른 Embedding을 활용해 학습하여 0.8642의 성능을 얻음
- Head Ensemble을 진행한 Transformer와 LGBM의 결과를 soft-voting하여 0.8616의 성능을 얻음

# Experiment

## EDA : 모델을 활용한 데이터의 패턴 분석
### GMF
- 유저 데이터가 적어 유저 임베딩을 사용하면 모델 과적합이 발생함
- 문항에 대한 변수를 추가할수록 모델의 성능이 향상됨
- https://github.com/boostcampaitech3/level2-dkt-level2-recsys-05/pull/2

### LightGCN
- 유저-문항을 그래프 형태로 표현하여, 유저-유저 연관성을 기반으로 데이터를 표현하면 모델 성능이 좋자 않다는 것을 확인함

### Item2Vec
- 유저의 문제 풀이 내역을 기반으로 문항을 임베딩 했을 때, 특정한 패턴이 존재한다는 것을 알 수 있음
- 문제 풀이 내역 마다 공통된 특정한 문항 풀이 패턴이 존재
- https://github.com/boostcampaitech3/level2-dkt-level2-recsys-05/issues/4

<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168410454-9b1c28e2-c499-47c1-9185-e84d29c80e68.png" /></p>
    
### BERT(양방향) VS Transformer(단방향)
- 문제 풀이 내역을 양방향으로 학습하는 것보다 단반향으로 학습을 하는 것이 더 좋은 성능을 보임
- 즉 유저의 문제 풀이 순서는 매우 중요한 패턴이 내재되어 있다는 것을 알 수 있음
- + Trasformer을 이용한 다음 문제 예측
    - 단순히 문항을 이용해 다음에 등장할 문제를 예측하더라도 좋은 정확도를 보임

### 정리
- 유저를 표현할 수 있는 데이터의 수는 매우 적음 -> 유저를 임베딩 하는 것보다 최대한 문항 정보를 활용하여 parameterized function을 만드는 것이 중요
- 유저의 문제 풀이 순서는 매우 중요한 패턴 → 시간적 순서를 효과적으로 표현할 수 있는 Model Architecture 설계가 중요

## Model Architecture : 데이터의 패턴을 효과적으로 표현할 수 있는  custom model 개발

<p align="center"><img src="https://user-images.githubusercontent.com/65529313/168410457-dae68e33-7618-4b5e-b732-40c3767deb30.png" /></p>

- 과거 풀이 정보와 현재 풀이 정보 간의 연관성을 표현
- 서로 다른 Embedding Layer를 두어 non-convex 한 목적 함수를 convex하게 만듬
- https://github.com/boostcampaitech3/level2-dkt-level2-recsys-05/pull/6

## Feature Engineering & Training Method : 일반화된 모델을 만들기 위한 학습 방법
- Representation Learning
    - 데이터의 표현을 효과적으로 학습할 수 있는 Model Architecture
    - 최소한의 변수로 최대의 성능
- Feature Selection
    - 범주형 - 문항, 시험지, 태그, 시험지 대분류, 시간, 요일 등
    - 숫자형 - 정답률의 평균 / 표준편차, 풀이 시간의 평균 / 표준 편차 등
- Loss
    - 전체 Time-step에 대하여 Loss를 계산하여 유저 데이터 부족 문제를 해결
- Padding
    - Max-len과 mean-len의 차이가 크기 때문에, 배치 마다 서로 다른 크기의 padding을 두어 모델을 학습

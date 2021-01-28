# Kaggle Competition
## Overview
**Competition :** Titanic

**Objective :** Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck
## Description
+ **EDA & Data Preprocessing**
  + PassengerID
     + 학습에 영향을 주지 않는 Feature이므로 Train과 Test Data에서 Drop.
  + Cabin
    + NULL 값이 많아 Survived와의 관계성을 찾거나, NULL 값이 아닌 데이터를 이용하여 NULL 값을 채워넣기 어려우므로 Cabin Feature도 Train과 Test에서 Drop.
  + Ticket
    + 학습을 위해 string 타입의 Ticket Feature를 int 타입으로 mapping을 해야 하지만, 숫자와 알파벳이 섞여있어 mapping 하기가 어려우므로 Train과 Test에서 Drop.
  + PClass
    +  Survived Feature와 관계성 분석을 위해 countplot을 이용. 분석 결과 좌석등급이 낮은 3rd Class에 사람이 많이 탑승하였고 좌석등급이 높은 좌석일수록 생존율이 높은 것을 알 수 있음.
  + Sex  
    + Null 값이 존재하지 않지만 학습을 위해 남자를 1로 여자를 0으로 mapping.
  + Embarked
    + Survived Feature와의 관계를 분석한 결과, S에서 탑승한 승객이 많았고 C에서 탑승한 승객들의 생존율이 높았음을 알 수 있었음. 
    + 2개의 NULL값이 존재하여, 승객들이 많이 탑승한 S 값으로 채워줌.
    + Sex Feature와 마찬가지로 학습을 위해 string을 int형으로 mapping 시켜줌.
  + Fare
    + Test Data에서 1개의 NULL값이 존재하므로 단순히 median 값으로 채워줌. 
    + 연속적인 숫자 값이기 때문에 범위를 정해주어 연속적이지 않게 바꿔주기 위해 5개의 범위로 나누어 주었음. 이때 Fare가 작은 값이 매우 많이 때문에 cut을 그대로 사용하게 되면 특정 범위 안에 너무 많은 값이 존재할 수 있으므로 qcut을 이용하여 백분위를 기준으로 나누어줌 그 후 범위마다 특정 숫자로 mapping을 하여 train과 test데이터 모두 Fare feature를 연속적이지 않게 바꿔준 다음 data type을 int형으로 casting 해줌.
  + SibSp & Parch
    + SibSp Feature를 countplot을 이용하여 분석한 결과, 혼자 탑승한 사람이 많았고 이들의 생존율은 낮았음을 알 수 있었음.
    + SibSp Feature와 Parch Feature는 가족이라는 공통적인 특징을 가지고 있으므로 두개의 Feature를 Family라는 하나의 Feature로 합침. 이때 본인을 포함해주기 위해 뒤에 1 값을 더해줌.
    + Family Feature를 countplot을 이용하여 분석한 결과, 혼자 탑승한 승객이 압도적으로 많았고 이들의 생존율 또한 낮았음을 알 수 있었음.
    + 혼자 탑승한 승객이 압도적으로 많으므로 혼자 탑승한 승객과 그렇지 않은 승객으로 값을 Binary하게 바꿔줌.
    + Survived Feature와의 관계성을 분석한 결과, 혼자 탑승한 승객의 생존율은 낮고 가족과 함께 탑승한 승객은 생존율이 상대적으로 높음을 알 수 있었음.
  + Age
    + Age에는 NULL 값이 많아 처음에는 성별에 따른 나이의 median값으로만 채워주려고 했지만 두 성별의 median값이 같아 모델이 학습하는데에 도움이 되지 않을 것으로 생각되어 Pclass feature를 추가하여 두개의 feature 값에 따른 median값으로 나이를 계산하였음.
    + null값들을 Pclass와 성별에 맞게 계산해둔 median값들로 채워넣어주었음
    + Fare와 마찬가지로 age또한 연속적인 값이기 때문에 연속적이지 않게만들어주기 위해 범위를 8개로 나누어 주고, 나눈 범위에 따라 특정값에 mapping을 해주었음.
  + Name
    + 외국 이름에서 공통적으로 들어가는 영어 호칭들에 따라 value값들을 count 해서 표로 표현해줌
    + Mlle, Ms, Mme는 각각 Miss와 Mrs의 프랑스식 표현이므로 같이 묶어 주고 나머지는 Rare로 묶어주었음.
    + string형이기 때문에 마찬가지로 int값으로 mapping을 시켜주고 train과 test에서 모두 알맞은 값으로 mapping 시켜줌.
+ **Model & Results**
  + 총 7가지 Model을 이용하여 생존자를 예측하였고, 각 Model의 LB Score는 다음과 같음
    + LogisticRegression  ➝  0.77033
    + Support Vector Machine ➝ 0.78468
    + KNN ➝ 0.77990
    + Decision Tree ➝ 0.75590
    + Random Forest ➝ 0.77511
    + AdaBoost ➝ 0.78947
    + Bagging ➝ 0.77033

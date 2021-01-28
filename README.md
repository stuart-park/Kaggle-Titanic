# Kaggle Competition
## Overview
**Competition :** Titanic

**Objective :** Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck
## Description
+ **Data Preprocessing**
  + PassengerID
     + PassengerId Feature 같은 경우 DataFrame에서 Index와 같은 역할을 하기 때문에 학습에 영향을 주지 않으므로 Train과 Test에서 Drop 시킴
  + Cabin
    + 위에서 나왔듯이 NULL 값이 너무 많아 Survived와의 관계성을 찾기가 힘들고,NULL값을 채워넣기 힘드므로, Cabin 값 또한 Train과 Test에서 Drop 시킴
  + Ticket
    + Ticket 같은 경우 위에서 DataFrame의 정보에서 봤듯이 object, 즉 string 형으로 이루어져 있음. 이 string데이터를 학습하기 위해서는 int value 형으로 바꿔줘야 하지만 데이터 표에서 볼 수 있듯이 숫자와 알파벳이 섞여 있으므로 어떠한 num 값으로 mapping하기가 힘드므로 이 Feature 또한 DataFrame에서 Drop 시킴
  + PClass
    + Pclass는 이미 int value type이고 null 값 또한 존재하지 않으므로 Survived feature와의 관계성을 찾아보기 위해 countplot을 이용해서 관계성을 찾아봄. 코드에서 볼 수 있듯이 좌석등급이 좋지 않은 3rd class에 사람이 많이 탑승하였고 좌석등급이 좋은 좌석일수록 생존율이 높은 것을 알 수 있음
  + Sex  
    + 성별은 Null 값이 존재하지 않지만 object형이므로 int형으로 mapping 시켜줘야 함. 남자를 1값으로 여자를 0값으로 mapping 시켜주고 train과 test모두에서 object값에 맞게 알맞은 int값으로 mapping 시켜줌
  + Embarked
    + Embarked은 배 탑승 장소를 나타내는 feature로써 이 feature 또한 countplot을 이용해서 Survived과의 관계성을 찾아봄. S에서 탑승한 승객이 많았고 C에서 탑승한 승객들의 생존율이 높았음을 알 수 있음. 데이터 불러오는 과정에서 embarked에 2개의 NULL값이 존재함을 알 수 있었고 이 값을 단순히 승객들이많이 탑승한 S 값으로 채워넣어줌. 그 후 성별과 마찬가지로 S, C, P 값을 int 값으로 mapping 해준 다음, dataframe에서 object값에 맞게 알맞은 int값으로 mapping 시켜줌
  + Fare
    + Test dataframe에서 Fare feature에서 1개의 NULL값이 존재하므로 단순히 median값으로 채워주었음. 그 후 fare band는 연속적인 숫자 값이기 때문에 범위를 정해주어 연속적이지 않게 바꿔주기 위해 5개의 범위로 나누어 주었음. 
    + Fare가 작은 값이 매우 많이 때문에 cut을 그대로 사용하게 되면 특정 범위 안에 너무 많은 값이 존재할 수 있으므로 qcut을 이용하여 백분위를 기준으로 나누어줌 그 후 범위마다 특정 숫자로 mapping을 하여 train과 test데이터 모두 Fare feature를 연속적이지 않게 바꿔준 다음 data type을 int형으로 casting 해줌.
  + SibSp & Parch
    + SibSp를 countplot을 이용해 분석해 보면 혼자 탑승한 사람이 많고 이들의 생존율은 낮았음을 알 수 있음
    + SibSp와 Parch feature는 모두 가족이라는 공통점을 가지고 있으므로 두개의 Feature를 Family라는 하나의 Feature로 합추어줌. Train과 Test에서 모두 SibSp와 Parch를 합쳐 Family라는 새로운 Feature를 생성해 주었음. 그리고 본인을 포함해주기 위해 뒤에 1 값을 더해주었음.
    + Family feature를 countplot을 이용하여 분석해보았을 때 혼자 탑승한 승객이 압도적으로 많았고 이들의 생존율은 낮았음을 알 수 있음.
    + 혼자 탑승한 승객이 압도적으로 많으므로 혼자 탑승한 승객과 그렇지 않은 승객으로 값을 바꿔주었음
    + 데이터 값들을 바꿔준 다음 다시 countplot을 이용하여 분석해보면 혼자 탑승한 승객들의 생존율은 낮고 가족과 함께 탑승한 승객은 생존한 승객이 조금 더 많음을 알 수 있음.
  + Age
    + Age에는 NULL 값이 많아 처음에는 성별에 따른 나이의 median값으로만 채워주려고 했지만 두 성별의 median값이 같아 모델이 학습하는데에 도움이 되지 않을 것으로 생각되어 Pclass feature를 추가하여 두개의 feature 값에 따른 median값으로 나이를 계산하였음.
    + 그 후 null값들을 Pclass와 성별에 맞게 계산해둔 median값들로 채워넣어주었음
    + 그 후 Fare와 마찬가지로 age또한 연속적인 값이기 때문에 연속적이지 않게만들어주기 위해 범위를 8개로 나누어 주고, 나눈 범위에 따라 특정값에 mapping을 해주었음.
  + Name
    + 외국 이름에서 공통적으로 들어가는 영어 호칭들에 따라 value값들을 count 해서 표로 표현해줌
    + Mlle, Ms, Mme는 각각 Miss와 Mrs의 프랑스식 표현이므로 같이 묶어 주고 나머지는 Rare로 묶어주었음.
    + 그리고 data type 이 string형이기 때문에 마찬가지로 int값으로 mapping을 시켜주고 train과 test에서 모두 알맞은 값으로 mapping 시켜줌.
+ **Model & Results**
  + 총 7가지의 모델을 이용하여 생존자를 예측하고 각 모델의 점수는 다음과 같음
    + LogisticRegression  ➝  0.77033
    + Support Vector Machine ➝ 0.78468
    + KNN ➝ 0.77990
    + Decision Tree ➝ 0.75590
    + Random Forest ➝ 0.77511
    + AdaBoost ➝ 0.78947
    + Bagging ➝ 0.77033

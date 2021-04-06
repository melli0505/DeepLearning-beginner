# 2. Logistic Regression

binary classification - 이진 분류 

## Perceptron

이진 분류 문제에서 최적의 가중치 학습

구조

- 선형회귀와 유사(직선방정식 사용)
- 마지막 단계에서 샘플 이진분류를 위해 step function 사용
- step function을 통과한 값을 가중치와 절편 학습에 사용

$w_1x_1 + w_2x_2 + b = z$ 를 선형함수라고 할 때,

계단함수는 z가 0보다 작으면 -1(음성 클래스)로, 크면 1(양성 클래스)로 분류한다. 

역방향 연산은 선형 함수 이후, step function 이후에 진행

## Multi-class

특성이 여러 개인 문제. 시그마 기호로 표기 가능

$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b = b + \sum w_ix_i$$

여러 개의 가중치를 가진다. n번째 특성의 가중치 = Wn

## 적응형 선형 뉴런, 아달린

퍼셉트론과 달리 계단함수의 결과를 가중치 업데이트에 사용하는 게 아니라 선형함수의 결과만 업데이트에 사용, 계단함수 결과는 예측에만 사용.

역방향 연산이 선형 함수 이후에 진행 ⇒ 개선버전 = 로지스틱 회귀

## Logistic Regression

선형함수 ⇒ 활성화함수의 결과를 역방향 연산에 사용. 이후에 step function(임계함수) 사용

활성화 함수 = 임계함수 전에 값을 변형시켜주기 위해서 사용. 비선형 함수

- 활성화 함수가 선형이라면 이전 선형함수와 곱해줘도 결과가 선형이므로 뉴런을 여러개 쌓아도 결국 선형.
- 따라서 활성화 함수(activation function)은 비선형이어야함.
- logistic regression 에서는 sigmoid 활성화 함수 사용

## Sigmoid

odds ratio ⇒ logit function ⇒ sigmoid 순으로 만들 수 있음

1. **odds ratio**
    - p가 성공확률 이라고 할 때,  $OR(OddsRatio)=p/(1-p)$
    - p가 0부터 1까지 증가할 때 처음에는 천천히 증가하다가 1에 가까워지면 급격히 증가

    ![logistic-regression\image_md\Untitled.png](logistic-regression\image_md\Untitled.png)

2. **logit function**
    - $logit(p) = log(\frac{P}{1-P})$
    - P가 0.5일 때 0, 0과 1일 때 각각 무한대로 음수와 양수가 된다.
    - 세로 축을 z라고 할 때, $z = log(\frac{P}{1-P})$

    ![logistic-regression\image_md\Untitled1.png](logistic-regression\image_md\Untitled1.png)

3. **sigmoid**
    - 위 식을 z에 대하여 정리

    $$p = \frac{1}{1 + e^{-z}}$$

    ![logistic-regression\image_md\Untitled2.png](logistic-regression\image_md\Untitled2.png)

이진 분류시에 z의 범위를 무한대에서 0~1 범위로 줄이기 위해 sigmoid 사용

⇒ z 값을 확률처럼 사용할 수 있다.

임계함수에서 z ≥ 0.5와 z ≤ 0.5를 기준으로 분류할 수 있다.

## Logistic cost function

다중 분류를 위한 cross entropy의 이진 분류 버전

$$L = -y\log(a) + (1-y)\log(l-a)$$

*️⃣ a = 활성화 함수의 출력값, y = target

y = 1이면 $-\log(a)$, y = 0이면 $-\log
(1-a)$

위 식을 최소로 만들 때, 양성 클래스의 경우에는 a=1에 수렴하고, 음성 클래스는 a=0에 수렴하므로 우리가 원하는 목표치와 같아진다. ⇒ 로지스틱 손실 함수를 최소화하면 이상적인 a 값이 나온다.

## Logistic cost function 미분

1. 가중치에 대한 편미분

    $$∂L/∂w = -(y - a)x_i$$

2. 절편에 대한 편미분

$$∂SE/∂w = -(y-a)1$$

제곱오차의 미분과 비교했을 때 $\hat y$만 a로 바뀌고 똑같다. (미분 과정 생략)

## 가중치/절편 업데이트

$$w_i = w_i - \frac{∂L}{∂w} = w_i + (y-a)x_i$$

$$b = b - \frac{∂L}{∂w} = b + (y-a)1$$

# 3주차 과제 답!

⇒ 데이터가 고른 범위를 갖고있지 않기 때문!

![logistic-regression\image_md\Untitled3.png](logistic-regression\image_md\Untitled3.png)

각각의 특성값에 대한 범위가 너어어무 달라서 데이터마다 값이 천차만별이라서 발생한 문제. 학습 중 오버피팅이 발생한 예시. 

**데이터 전처리**를 통해서 해결할 수 있다.

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
y = cancer.target

# 속성별 평균
m = cancer.data.mean(axis=0)

# 속성별 표준편차
s = cancer.data.std(axis=0)

# 정규 데이터
x = (cancer.data - m)/s
```

x 값을 위와 같이 넣어서 바꿔준 다음에 다시 그래프를 출력해보면

![logistic-regression\image_md\Untitled4.png](logistic-regression\image_md\Untitled4.png)

얼추 비슷한 범위 안에 값들이 모여있는 것을 확인할 수 있다.

### 데이터 전처리 과정을 거친 후에 다시 돌려보고 결과와 그래프 비교

- 이전

![logistic-regression\image_md\Untitled5.png](logistic-regression\image_md\Untitled5.png)

- 이후

![logistic-regression\image_md\Untitled6.png](logistic-regression\image_md\Untitled6.png)

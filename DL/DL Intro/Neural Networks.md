[Источник](https://cs231n.github.io/neural-networks-1/)
## Modeling one neuron
![[Снимок экрана 2026-05-26 в 23.06.23.png|352]]
>_softmax_ - превращаем набор чисел в набор вероятностей
> $\mathcal{L}_{CE} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$

### Single neuron as a linear classifier

**Binary Softmax classifier**. For example, we can interpret σ(∑w<sub>i</sub>x<sub>i</sub>+b) to be the probability of one of the classes P(yi=1∣xi;w). The probability of the other class would be P(yi=0∣xi;w)=1−P(yi=1∣xi;w). With this interpretation, we can formulate the cross-entropy loss and optimizing it would lead to a binary Softmax classifier (_logistic regression_)

**Binary SVM classifier**. Alternatively, we could attach a max-margin hinge loss to the output of the neuron and train it to become a binary Support Vector Machine.

### Commonly used activation functions

**Sigmoid:**  σ(x) = 1/(1+e<sup>−x</sup>)
- (-) _Sigmoids saturate and kill gradients_.
- (-) _Sigmoid outputs are not zero-centered_.
**Tanh**
-  _tanh(x)=2σ(2x)−1._
-  (+) _zero-centered_
**ReLU**:  f(x) = max(0, x)
- (+) _It was found to greatly accelerate the convergence of stochastic gradient descent compared to the sigmoid/tanh functions_
- (+) _ReLU проще имплементировать_
- (-) _ReLU units can be fragile during training and can “die”_
**Leaky ReLU**:  f(x)=𝟙(x<0)(αx)+𝟙(x>=0)(x)
- (+) _attempt to fix the “dying ReLU” problem_
**Maxout**:  max(w<sup>T</sup><sub>1</sub>x+b<sub>1</sub>, w<sup>T</sup><sub>2</sub>x+b<sub>2</sub>,)
- (+) _все плюсы leaky ReLU_
- (+) _нет “dying ReLU” problem_
- (-) _doubles the number of parameters_

## Neural Network architectures
### Layer-wise organization
![[Снимок экрана 2026-05-27 в 12.28.11.png]]
### Example feed-forward computation

_One of the primary reasons that Neural Networks are organized into layers is that this structure makes it very simple and efficient to evaluate Neural Networks using matrix vector operations._

```python
# forward-pass of a 3-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x))
x = np.random.randn(3, 1)
h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)
```
### Representational power

> Neural Networks with at least one hidden layer are _universal approximators_
- [Интуиция](http://neuralnetworksanddeeplearning.com/chap4.html)

> Given any continuous function f(x) and some ϵ > 0, there exists a Neural Network g(x) with one hidden layer (with a reasonable choice of non-linearity, e.g. sigmoid) such that ∀x: ∣f(x)−g(x)∣ < ϵ.

**Why use more layers and go deeper?**
- Глубокая сеть может выразить ту же зависимость гораздо компактнее.
- The fact that deeper networks (with multiple hidden layers) can work better than a single-hidden-layer networks is an empirical observation, despite the fact that their representational power is equal.

### Setting number of layers and their sizes

First, note that as we increase the size and number of layers in a Neural Network, the **capacity** of the network increases. That is, the space of representable functions grows

![[Pasted image 20260527132846.png|497]]

However, this is both a blessing (since we can learn to classify more complicated data) and a curse (since it is easier to overfit the training data). But in practice, it is always better to use other methods to control overfitting instead of the number of neurons (smaller networks are harder to train with local methods such as Gradient Descent).

![[Pasted image 20260527133507.png|497]]

## Setting up the data and the model
[Источник](https://cs231n.github.io/neural-networks-2/)
### Data Preprocessing

**Mean subtraction**: `X -= np.mean(X, axis = 0)`
**Normalization**: `X /= np.std(X, axis = 0)`

**[Principal Component Analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) and Whitening**: 
1.  Mean subtraction
2. `cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix`
3. `U, S, V = np.linalg.svd(cov)`
4. Чтобы устранить корреляцию в данных, мы проецируем исходные (но центрированные по нулю) данные на собственный базис: `Xrot = np.dot(X, U)`
5. `Xrot_reduced = np.dot(X, U[:,:100])` (*PCA* именно тут, св-ва SVD); X: [N x D] -> [N x 100], keeping the 100 dimensions of the data that contain the most variance
6. *Whitening* operation takes the data in the eigenbasis and divides every dimension by the eigenvalue to normalize the scale. The geometric interpretation of this transformation is that if the input data is a multivariable gaussian, then the whitened data will be N(0, 1). `Xwhite = Xrot / np.sqrt(S + 1e-5)`

![[Pasted image 20260527135847.png]]

### Weight Initialization

**Pitfall: all zero initialization**: bad ( because if every neuron in the network computes the same output, then they will also all compute the same gradients during backpropagation and undergo the exact same parameter updates)
**Small random numbers**: `W = 0.01 * np.random.randn(D,H)`

>[!info] Calibrating the variances with 1/sqrt(n). The distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that we can normalize the variance of each neuron’s output to 1 by scaling its weight vector by the square root of its _fan-in_ (i.e. its number of inputs). 

Итог: `w = np.random.randn(n) / sqrt(n)`.
Но для ReLU лучше так: `w = np.random.randn(n) * sqrt(2.0/n)`
**bias** можно инициализировать нулями, потому что симметрия уже нарушается случайной инициализацией весов.

**Batch Normalization** - слой нейросети, который после линейного преобразования нормализует активации по текущему батчу

![[Снимок экрана 2026-05-27 в 14.52.24.png|328]]

### Regularization

- **L1 regularization:**  $\mathcal{L}_{\text{L1}} = \mathcal{L}(\theta) + \lambda \|\theta\|_1$ (имеет склонность занулять веса)
- **L2 regularization:**  $\mathcal{L}_{\text{L2}} = \mathcal{L}(\theta) + \lambda \|\theta\|_2^2$ 
- **Elastic Net regularization:**  $\mathcal{L}_{\text{ElasticNet}} = \mathcal{L}(\theta) + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2$
- **Max norm constraints:**  $\|\theta\|_2 < c$
- **Dropout:** while training, dropout is implemented by only keeping a neuron active with some probability **p**, or setting it to zero otherwise.

```Python
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""
p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # (* p) здесь нет, мы его засунули в train, поэтому это inverted dropout
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```

**Bias** обычно не регуляризуют, потому что он не управляет влиянием признаков посредством умножения.
**Per-layer regularization**. It is not very common to regularize different layers to different amounts
**In practice**: It is most common to use a single, global L2 regularization strength that is cross-validated. It is also common to combine this with dropout applied after all layers. The value of p=0.5 is a reasonable default, but this can be tuned on validation data.

### Loss functions

**Classification:** 
- SVM loss:  $L_i^{\text{SVM}} =\sum_{j \neq y_i}\max \left(0, f_j - f_{y_i} + \Delta \right)$
- Cross-Entropy loss: $L_i^{\text{CE}} =-\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right)$
**Attribute classification:**
- $L_i = \sum_j \max \left(0,\; 1 - y_{ij} f_j \right), \quad y_{ij}=+1 \text{ если атрибут } j \text{ есть,} \quad y_{ij}=-1 \text{ если его нет.}$
- Или можно учить лог. регрессию для каждого атрибута по отдельности
- Log-loss = negative log-likelihood: $L_i = -\sum_j \left[y_{ij}\log(\sigma(f_j)) + (1-y_{ij})\log(1-\sigma(f_j))\right]$
**Regression:**
- MAE, MSE, RMSE, MAPE, ...

## Learning
[Источник](https://cs231n.github.io/neural-networks-3/)

**Gradient Checks**

$\frac{df(x)}{dx} = \frac{f(x+h)-f(x)}{h} \quad$ (bad, do not use) $\quad\quad\quad\quad\frac{df(x)}{dx} = \frac{f(x+h)-f(x-h)}{2h} \quad$ (use instead)

### Parameter updates

**Vanilla update:**  `x += - learning_rate * dx`
Далее будет подробнее про оптимизаторы (из курса Методы Выпуклой оптимизации)

![[opt1.gif|280]] ![[opt2.gif|280]]

### Общий подход (вариант 1)

1. **Прямой проход**:   $\hat{y} = f(x; \theta)$
2. **Вычисление потерь**: $\mathcal{L} = \text{loss}(\hat{y}, y)$
3. **Обратный проход**: вычислить $\nabla_\theta \mathcal{L}$ через backpropagation
4. **Обновление весов**: $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$ (или через более сложный оптимизатор) 

### Общий подход (вариант 2)

 - Параметрическая модель: $f(x, \theta)$
 - Обучающая выборка: $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}$
 - Функция потерь: $\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell\bigl(f(x_i;\theta),y_i\bigr)$      *(эмпирический риск)*
 - Ищем $\theta^{*}=\arg\min_{\theta}\mathcal{L}(\theta)$
### Дополнительно

- **Feedforward Neural Network (FFN)** — это нейросеть, в которой информация передается только от входа к выходу, без обратных связей и циклов.
- **MLP (Multilayer Perceptron)** — частный случай FFN: многослойная полносвязная сеть.
- **Mini-batch training** — это способ обучения нейросети, когда данные делят на небольшие группы, и модель обновляет параметры после обработки каждой такой группы
### Автоматическое дифференцирование

![[Автоматическое дифференцирование.pdf]]

**Forward-mode и reverse-mode** не конкурируют, а дополняют друг друга.

Если входов мало, а выходов много, часто удобнее forward-mode. Если входов много, а выход скалярный — reverse-mode. Именно поэтому для обучения больших моделей reverse-mode почти безальтернативен.
[cs231 book](https://cs231n.github.io/convolutional-networks/) [Andrej Karpathy lecture](https://www.youtube.com/watch?v=u6aEYuemt0M)

ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

### Architecture Overview

Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3, so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. It will lead to overfitting.

3D volumes of neurons. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth.

![[Pasted image 20260601203656.png|272]]![[Pasted image 20260601203711.png|376]]
>A ConvNet is made up of Layers. Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.

### Layers used to build ConvNets

We use three main types of layers to build ConvNet architectures: **Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet **architecture**.

Example Architecture: ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:
- INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
- CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
- RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
- POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
- FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score.

#### Convolutional Layer

Свёрточный слой состоит из набора обучаемых фильтров: каждый фильтр имеет небольшие размеры по ширине и высоте, но проходит через всю глубину входного объёма. Во время прямого прохода фильтр скользит по ширине и высоте входных данных, вычисляет скалярные произведения в каждой позиции, создаёт двумерную карту активаций, а затем карты от всех фильтров складываются по глубине и образуют выходной объём.

_Example 1_. For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of 75 + 1 weights. Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

![[Pasted image 20260601210054.png|266]]![[Pasted image 20260601210102.png|316]]

We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven’t yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the **depth, stride** and **zero-padding**.

1. First, the **depth** of the output volume is a hyperparameter: it corresponds to the number of filters(=kernels) we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color.
2. Second, we must specify the **stride** with which we slide the filter. When the stride is 1 then we move the filters one pixel at a time и тд.
3. As we will soon see, sometimes it will be convenient to pad the input volume with zeros around the border. The size of this **zero-padding** is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes. Но обычно заполняют не нулями

We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. The correct formula for calculating how many neurons “fit” is given by (W−F+2P)/S+1. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output. Lets also see one more graphical example: (S = 1 и S = 2)

![[Pasted image 20260601210943.png]]

**Summary**. To summarize, the Conv Layer:

- Accepts a volume of size W1×H1×D1
- Requires four hyperparameters:
    - Number of filters K,
    - their spatial extent F,
    - the stride S,
    - the amount of zero padding P.
- Produces a volume of size W2×H2×D2 where:
    - W2=(W1−F+2P)/S+1
    - H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
    - D2=K
- With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
- In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.

![[conv_demo.gif]]
![[Снимок экрана 2026-06-03 в 22.38.48.png|381]]

- **Локальность** - нейрон в свёрточном слое смотрит не на всё изображение, а только на маленький участок, например 3×3 или 5×5. Это позволяет искать локальные признаки: края, углы, текстуры.
- **Разделение весов / weight sharing** - один и тот же фильтр применяется ко всем позициям входа. Благодаря этому сеть ищет один и тот же признак в разных местах изображения и использует намного меньше параметров.
- **Padding** - добавление нулей по краям входа. Нужно, чтобы контролировать размер выхода и не терять информацию на границах изображения.
- **Stride** - шаг, с которым фильтр двигается по входу.  
- **Dilation** - “растяжение” фильтра: между элементами ядра появляются промежутки. Это увеличивает область обзора фильтра без увеличения числа параметров.
- **Receptive field** - участок исходного входа, который влияет на конкретный элемент выходной карты признаков. Чем глубже слой, больше kernel/stride/dilation - тем больше receptive field.

**1x1 convolution** - это свёртка с ядром размером 1×1, которая обрабатывает каждый пиксель отдельно, но смешивает информацию между каналами. Её часто используют, чтобы уменьшить или увеличить число каналов, например из 256 сделать 64, не меняя высоту и ширину карты признаков.

**Depthwise convolution** и **depthwise-separable convolution**: [6-минутное объяснение](https://www.youtube.com/watch?v=vVaRhZXovbw)
![[Снимок экрана 2026-06-02 в 14.17.32.png|333]]![[Снимок экрана 2026-06-02 в 14.19.09.png|339]]
depthwise-separable = depthwise + 1x1

**Transposed convolution**: [7-минутное объяснение, но поймете за первые 3](https://www.youtube.com/watch?v=qb4nRoEAASA)

#### Pooling Layer

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting

More generally, the pooling layer:
- Accepts a volume of size W1×H1×D1
- Requires two hyperparameters:
    - their spatial extent F (обычно = 2/3),
    - the stride S (обычно = 2),
- Produces a volume of size W2×H2×D2 where:
    - W2=(W1−F)/S+1
    - H2=(H1−F)/S+1
    - D2=D1
- Introduces zero parameters since it computes a fixed function of the input
- For Pooling layers, it is not common to pad the input using zero-padding.

Обычно используется MAX, но можно среднее или L2-норма

![[Pasted image 20260602131158.png|248]]![[Pasted image 20260602131205.png|401]]

### ConvNet Architectures

We have seen that Convolutional Networks are commonly made up of only three layer types: CONV, POOL and FC (short for fully-connected, **classifier head**). We will also explicitly write the RELU activation function as a layer, which applies elementwise non-linearity. In this section we discuss how these are commonly stacked together to form entire ConvNets.

#### Layer Patterns

The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. In other words, the most common ConvNet architecture follows the pattern:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

#### Layer Sizing Patterns

The **input layer** - степень двойки
The **conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of S=1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input.
The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2).

### Классические CNN-архитектуры.

#### LeNet 
The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990’s. Of these, the best known is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture that was used to read zip codes, digits, etc.
![[Снимок экрана 2026-06-02 в 14.33.57.png]]
![[Pasted image 20260602145054.png|448]]

```python
class LeNet(nn.Module):  
	def __init__(self, num_classes=10):  
		super().__init__()  
		self.conv1 = nn.Conv2d(  
			in_channels=1,  
			out_channels=6,  
			kernel_size=5  
		)  
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  
		self.conv2 = nn.Conv2d(  
			in_channels=6,  
			out_channels=16,  
			kernel_size=5  
		)  
		self.conv3 = nn.Conv2d(  
			in_channels=16,  
			out_channels=120,  
			kernel_size=5  
		) 
		self.fc1 = nn.Linear(120, 84)  
		self.fc2 = nn.Linear(84, num_classes)  
  
def forward(self, x):  
	x = torch.tanh(self.conv1(x)) # [batch_size, 6, 28, 28]  
	x = self.pool(x) # [batch_size, 6, 14, 14]  
	  
	x = torch.tanh(self.conv2(x)) # [batch_size, 16, 10, 10]  
	x = self.pool(x) # [batch_size, 16, 5, 5]  
	  
	x = torch.tanh(self.conv3(x)) # [batch_size, 120, 1, 1]  
	x = torch.flatten(x, start_dim=1) # [batch_size, 120]  
	  
	x = torch.tanh(self.fc1(x)) # [batch_size, 84]  
	x = self.fc2(x) # [batch_size, 10]  
  
	return x
```


#### AlexNet
The AlexNet was submitted to the [ImageNet ILSVRC challenge](http://www.image-net.org/challenges/LSVRC/2014/) in 2012 and significantly outperformed the second runner-up. The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer). Была использована ReLU, аугментация, а еще это первый перенос на GPU, первое использование dropout. 
![[Снимок экрана 2026-06-02 в 15.19.25.png]]
>*Изображение → Conv → Pool → Conv → Pool → Conv → Conv → 
>		→ Conv → Pool → FC → FC → FC → Класс*

В следующем году (2013) нашелся подход лучше - ZFNet, основное отличие: первый сверточный слой другой: (11x11, stride 4) -> (7x7, stride 2).

#### VGGNet
The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Its main contribution was in showing that the depth of the network is a critical component for good performance. Использовала только фильтры 3x3, stride = 1 и pooling 2x2. Очень простая и однородная архитектура.
![[Pasted image 20260602154029.png|543]]![[Pasted image 20260602154145.png|144]]
A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M)
 
#### Inception
Inception - это семейство архитектур CNN. Самая известная ранняя версия - **GoogLeNet**, победившая в ImageNet 2014. Главная идея Inception: вместо того чтобы выбирать один размер свёртки, сеть параллельно применяет несколько разных операций к одним и тем же признакам, а потом объединяет результат.

**GoogLeNet**'s main contribution was the development of an _Inception Module_ that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently [Inception-v4](http://arxiv.org/abs/1602.07261).

Модуль Inception:
![[Pasted image 20260602170342.png|538]]
На первый взгляд, это параллельная комбинация свёрточных фильтров 1х1, 3х3 и 5х5. Но изюминка заключалась в использовании свёрточных блоков 1х1 (NiN) для уменьшения количества свойств перед подачей в «дорогие» параллельные блоки.

LeNet заложил базовую схему CNN - свёртки, pooling и классификатор, AlexNet показал силу глубоких сетей на GPU и ReLU, VGG систематизировала идею увеличения глубины через маленькие 3×3 свёртки. Inception развила feature extraction шире: параллельные свёртки разных масштабов позволили извлекать признаки разной сложности эффективнее, чем просто наращивать глубину.


#### ResNet
[Residual Network](http://arxiv.org/abs/1512.03385) developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special _skip (residual) connections_ and a heavy use of [batch normalization](http://arxiv.org/abs/1502.03167). The architecture is also missing fully connected layers at the end of the network.

**Skip Connections**.
- Skip connections позволяют пропускать информацию через один или несколько слоев. Эти связи позволяют передавать прямую информацию от одного слоя к следующему, минуя промежуточные слои, что помогает избежать затухания градиента и улучшает обучение очень глубоких сетей.
**Residual Blocks**
- Основным строительным блоком ResNet является резидуальный блок. Этот блок включает в себя несколько сверточных слоев с функцией активации ReLU, к которым добавляется прямое соединение, проходящее через весь блок
![[Pasted image 20260602172144.png|333]]
ResNet позволила строить сети с очень большим числом слоев, например, ResNet-50, ResNet-101 и ResNet-152 (и другие модификации), где цифры обозначают количество слоев в сети. Эти сети значительно глубже, чем предыдущие архитектуры, такие как VGG, и при этом обучаются эффективно благодаря резидуальным связям

**Сокращенные резидуальные блоки (Bottleneck Blocks)**
- В более глубоких версиях ResNet (например, ResNet-50 и глубже) используется концепция **bottleneck‑блоков**, где резидуальный блок состоит из трех сверточных слоев: сначала 1×1 для сокращения размерности, затем 3×3 для свертки, и снова 1×1 для увеличения размерности обратно. Это снижает количество вычислений и параметров при сохранении высокой производительности.

#### Идеи эффективных архитектур
хз что тут

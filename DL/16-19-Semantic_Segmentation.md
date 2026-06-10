Это задача компьютерного зрения, где каждому пикселю изображения присваивается класс объекта или область.
<img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2017.41.16.png" width="163"><img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2017.41.47.png" width="162"><img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2017.42.30.png" width="158"><img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2017.43.02.png" width="160">
**Очевидный подход:** скользящим окном проходить и каждый раз запускать CNN (очевидно плохо)
**Convolutions End-to-End:** через Conv слои, но это дорого, оставляя размер таким же
![](../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2018.05.47.png)
**Fully Convolutional Nets:** состоят только из conv слоёв без fully connected слоёв, уменьшаем изображение, но как увеличить?
<img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2018.11.21.png" width="394"><img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2018.14.11.png" width="203">

- *Bilinear Interpolation:* просто размываем, есть также cubic, bicubic, 2D-NN, 1D-NN

- *Deconvolution* (Transposed convolution из конспекта по CNN): [объяснение](https://www.youtube.com/watch?v=qb4nRoEAASA)
	 ![](../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2018.22.51.png)
	- Так расширение изображения происходит за счет обучения, в целом так получается *encoder-decoder подход*.

### U-Net
Combines all of the previous improvements but adds skip-connections between resolution levels to retrain high-res info

**skip-connections** - это связи в нейросети, которые “перепрыгивают” через один или несколько слоёв и передают входной сигнал дальше напрямую (восстанавливаем информацию потерянную при pooling)

 <img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2018.32.32.png" width="541">
 [подробное объяснение](https://www.youtube.com/watch?v=oxcgx75k6yU)
 
**Multi-scale features** - это признаки, извлечённые на разных масштабах/разрешениях, чтобы модель одновременно учитывала мелкие детали и крупный контекст изображения.
 
### Link-Net
Если U‑Net часто ассоциируется с конкатенацией skip‑признаков (и иногда довольно тяжёлым декодером), то LinkNet делает ставку на:
- encoder на ResNet,
- skip connections через сложение (summation),
- компактные decoder‑блоки.
<img src="../attachments/Pasted%20image%2020260603185924.png" width="603">

### Transformer-based segmentation
Vision Transformer [супер подробно](https://www.youtube.com/watch?v=rdIySBWpCHc)
Однако проще сначала почитать про трансформеры, а затем переходить к этой теме.
Transformer-сегментация использует механизм внимания, чтобы учитывать глобальные связи между участками изображения и лучше распознавать объекты сложной формы.
### Promptable segmentation
То есть модель получает изображение и **prompt** - подсказку, что именно нужно выделить и возвращает область пикселей, принадлежащих нужному объекту.


### Другое

**Panoptic segmentation.**  
Идея - разметить каждый пиксель изображения, объединяя semantic segmentation и instance segmentation.

**Human pose estimation.**  
Идея - найти ключевые точки тела человека: голову, плечи, локти, кисти, колени и другие суставы. По этим точкам строится скелет позы, который показывает положение человека и может использоваться для анализа движения, жестов или действий.

**Instance segmentation.**  
Идея - найти каждый отдельный объект на изображении и построить для него точную маску пикселей. В отличие от object detection, модель не просто рисует bounding box, а отделяет, например, одного человека от другого на уровне пикселей.

### Метрики и функции потерь

- *Pixel-wise cross-entropy (функция потерь):*
	- многоклассовая cross-entropy, применённая к каждому пикселю и затем усреднённая, если классы несбаланированны, то вводят веса.
	- $L_{\mathrm{CE}} = -\frac{1}{|\Omega|} \sum_{i \in \Omega} \log p_{i,y_i}.$

- *Intersection over Union (IoU):*
	- <img src="../attachments/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-06-03%20%D0%B2%2019.16.14.png" width="198">
	- $\mathrm{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$
	- $\mathrm{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \mathrm{IoU}_c$

- Mean Average Precision (mAP):
	- Это среднее качество модели по всем классам: насколько хорошо она находит нужные объекты и не добавляет лишние. Задается через порог совпадения рамок IoU, оцениваем через PR-curve.

- *Dice/F1:* 
	- $\mathrm{Dice} = \frac{2TP}{2TP + FP + FN}$
	- $\mathrm{Dice} = \frac{2\,\mathrm{IoU}}{1 + \mathrm{IoU}}$


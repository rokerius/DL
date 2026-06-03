[cs231 Transfer Learning](https://cs231n.github.io/transfer-learning/)

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. The three major Transfer Learning scenarios look as follows:
- **ConvNet as fixed feature extractor**
  Берём предобученную CNN, убираем её последний классификатор, замораживаем остальные слои и используем их как генератор числовых признаков изображения, поверх которых обучаем новый простой классификатор для своей задачи.
- **Fine-tuning the ConvNet**
  The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed and only fine-tune some higher-level portion of the network.
- **Pretrained models**

### When and how to fine-tune?

How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset. Here are some common rules of thumb for navigating the 4 major scenarios:
1. **New dataset is small and similar to original dataset.**
   *The best idea might be to train a linear classifier on the CNN codes.*
2. **New dataset is large and similar to the original dataset.**
   *Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.*
3. **New dataset is small but very different from the original dataset.**
   *Since the dataset is very different, it might not be best to train the classifier form the top of the network. It might work better to train the SVM classifier from activations somewhere earlier in the network.*
4. **New dataset is large and very different from the original dataset.**
   *Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.*

[Практика](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

FPN


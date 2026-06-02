#### Нейросети [тык](/Users/rokerius/Obsidian/Study/Study/DL/DL Intro/Neural Networks.md)
1. Введение в Deep Learning. Чем глубокое обучение отличается от классического ML, зачем нужны нейросети, какие типы данных и задач они хорошо обрабатывают.
2. Нейронная сеть как композиция функций. Линейный слой, bias, функции активации, глубина, нелинейность, softmax.
3. Обучение нейронной сети. Функция потерь, эмпирический риск, mini-batch training, train/validation/test split.
4. Backpropagation и автоматическое дифференцирование. Вычислительный граф, правило цепочки, forward-mode и reverse-mode autodiff.
#### Оптимизаторы [тык](/Users/rokerius/Obsidian/Study/Study/Convex Optimization/Оптимизаторы. DL Intro.md)
5. Оптимизация в DL. Gradient descent, SGD, Momentum, Nesterov, RMSProp, Adam/AdamW, роль learning rate.

6. Стабилизация и регуляризация обучения. Инициализация, нормализация, dropout, weight decay, early stopping, data augmentation.

#### PyTorch [тык](obsidian://open?vault=Study&file=DL%2FDL%20Intro%2FPyTorch)
7. Практический PyTorch-пайплайн. Tensor, autograd, nn.Module, Dataset, DataLoader, train/eval.

#### Нейросети (в конце) [тык](/Users/rokerius/Obsidian/Study/Study/DL/DL Intro/Neural Networks.md)
8. Диагностика обучения нейросети. Переобучение, leakage, tiny-overfit test, анализ train/validation curves, подбор гиперпараметров.

#### CNN [тык](obsidian://open?vault=DL&file=DL%2FDL%20Intro%2FConvolutional%20Neural%20Networks)
9. Изображение как тензор и мотивация CNN. Почему полносвязная сеть плохо использует структуру изображения.
10. Операция свёртки. Локальность, разделение весов, padding, stride, dilation, receptive field.
11. Базовая CNN для классификации изображений. Feature maps, conv, pooling, classifier head.
12. Виды свёрток и эффективные операции. 1x1 convolution, separable/depthwise convolution, transposed convolution.
13. Классические CNN-архитектуры. LeNet, AlexNet, VGG, Inception: как развивались идеи глубины, свёрток и feature extraction.
14. ResNet и современные CNN-backbone. Skip connections, residual block, bottleneck, **идеи эффективных архитектур**.

#### Transfer Learning [тык](obsidian://open?vault=DL&file=DL%2FDL%20Intro%2FTransfer%20Learning)
- Transfer learning в компьютерном зрении. Feature extraction, fine-tuning, заморозка слоёв, использование предобученных backbone.

- Семантическая сегментация. Постановка задачи, отличие от классификации, superpixels, FCN, encoder-decoder подход.
- U-Net и современные подходы к сегментации. Skip connections, upsampling, multi-scale features, transformer/promptable segmentation как расширения.
- Instance, panoptic segmentation и human pose estimation. Чем эти задачи отличаются от semantic segmentation, какие идеи используются.
- Метрики и функции потерь в сегментации. Pixel-wise cross-entropy, IoU/mIoU, Dice/F1, особенности оценки качества.
- Object detection. Постановка задачи, bounding boxes, confidence, IoU, отличие detection от classification и segmentation.

  

- Метрики и post-processing в детекции. Precision/recall, AP/mAP, Non-Maximum Suppression.
- Two-stage object detection. Sliding window, R-CNN, Fast R-CNN, Faster R-CNN, Region Proposal Network.
- One-stage object detection. YOLO, SSD, RetinaNet, anchors, anchor-free идеи, trade-off скорости и качества.
- Knowledge distillation. Teacher-student схема, soft labels, temperature, KL-divergence, зачем нужна дистилляция.
- Варианты и применения дистилляции. Дистилляция логитов, признаков, attention, online/offline/self-distillation, dataset distillation.
- Векторные представления слов. One-hot, distributional hypothesis, count-based embeddings, co-occurrence matrix, PMI/PPMI, LSA.
- Word2Vec и GloVe. Skip-gram, CBOW, negative sampling, интерпретация embedding space, аналогии и косинусная близость.
- Рекуррентные нейронные сети. Последовательные данные, hidden state, parameter sharing, BPTT, many-to-one/many-to-many задачи.
- Проблемы RNN и gated-архитектуры. Vanishing/exploding gradients, LSTM, GRU, teacher forcing, scheduled sampling. 
- Seq2seq и attention до трансформеров. Encoder-decoder, bottleneck одного вектора, attention для машинного перевода.
- Токенизация текста. Word-level, char-level, subword, Unicode/UTF-8, BPE, WordPiece, byte-level BPE.
- Механизм внимания. Query, Key, Value, scaled dot-product attention, multi-head attention.
- Архитектура Transformer. Self-attention, masked attention, cross-attention, FFN, residual connections, LayerNorm, positional encoding.
- Pretraining в NLP. Зачем нужен self-supervised pretraining, encoder-only, encoder-decoder и decoder-only подходы.
- BERT-подобные и T5-подобные модели. Masked language modeling, text-to-text постановка, fine-tuning под downstream tasks.
- GPT-подобные модели. Autoregressive generation, next-token prediction, decoder-only архитектура, in-context learning.
- Масштабирование трансформеров и MoE. Scaling laws, compute/data/model size, sparse experts, router, балансировка экспертов.
- Дообучение LLM. Full fine-tuning, supervised fine-tuning, instruction tuning, PEFT, adapters, LoRA, prefix/prompt tuning.
- Preference tuning и RLHF. Reward model, человеческие предпочтения, отличие RLHF от SFT, роль KL-регуляризации.
- In-context learning, prompting и Chain-of-Thought. Few-shot prompting, reasoning prompts, ограничения промптового подхода.
- Catastrophic forgetting при дообучении. Почему возникает, чем опасен, какие общие способы смягчения используются.
- Инференс LLM. Training vs inference, autoregressive generation, latency, KV-cache, почему генерация дорогая.
- Ускорение и сжатие LLM на инференсе. Efficient attention, квантизация INT8/INT4, trade-off памяти, скорости и качества.
- Декодирование LLM. Greedy decoding, sampling, temperature, top-k/top-p, beam search, связь стратегии декодирования с качеством ответа.
- RAG, длинный контекст и агенты. Как добавлять внешнее знание без переобучения, retrieval, tools, ограничения агентных схем.
- Звук как сигнал. Waveform, sampling, quantization, PCM, теорема Найквиста-Шеннона, aliasing.
- Частотное представление звука. Fourier transform, DFT/FFT, STFT, спектрограмма, mel-scale, log-mel features, MFCC.
- Automatic Speech Recognition. Постановка STT-задачи, уровни токенизации, WER/CER, alignment problem.
- CTC для распознавания речи. Blank token, collapse operation, CTC loss, greedy/beam decoding.
- Архитектуры ASR и TTS. RNN-T/LAS как альтернативы CTC; TTS pipeline: text normalization, acoustic model, vocoder.
- Vision Transformer. Patch embedding, positional encoding, self-attention для изображений, сравнение ViT и CNN.
- GAN. Генератор, дискриминатор, adversarial objective, min-max игра, основные проблемы обучения.
- Варианты и стабилизация GAN. Mode collapse, DCGAN, conditional GAN, pix2pix, CycleGAN, WGAN/WGAN-GP.
- Autoencoders и VAE. Reconstruction, bottleneck, latent space, ELBO, KL-divergence, reparameterization trick.
- Normalizing Flows и дискретные латентные модели. Обратимые преобразования, change of variables, exact likelihood, идея VQ-VAE.
- Графы как данные и GNN. Node/edge/graph-level задачи, message passing, агрегация соседей, примеры GCN/GraphSAGE/GAT.
- Проблемы и современные направления GNN. Oversmoothing, oversquashing, heterophily, масштабирование и graph foundation models.
- Диффузионные модели. Генерация из шума, score matching, denoising diffusion, forward/reverse process, U-Net, timestep embeddings.
- Условная и ускоренная диффузия. Guidance, text-to-image, DDIM/solvers/distillation, inpainting, flow matching.
- Мультимодальные задачи и CLIP. Image-text retrieval, zero-shot classification, contrastive learning, общее пространство изображений и текста.
- Vision-Language Models и MLLM. Captioning, VQA, OCR, BLIP/LLaVA-подходы, visual encoder, projector, LLM, visual tokens.
- Оценка и ограничения мультимодальных моделей. Бенчмарки, работа с большими изображениями, multi-image режим, типичные ошибки VLM.
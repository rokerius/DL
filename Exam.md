# Критерии оценивания устного экзамена по Deep Learning

Формат: устный экзамен, билет + дополнительные вопросы, оценка по 10-балльной шкале

## Список вопросов к устному экзамену

1. Введение в Deep Learning. Чем глубокое обучение отличается от классического ML, зачем нужны нейросети, какие типы данных и задач они хорошо обрабатывают.
2. Нейронная сеть как композиция функций. Линейный слой, bias, функции активации, глубина, нелинейность, softmax. [Neural_Networks](DL/Neural_Networks.md)
3. Обучение нейронной сети. Функция потерь, эмпирический риск, mini-batch training, train/validation/test split. [Neural_Networks](DL/Neural_Networks.md)
4. Backpropagation и автоматическое дифференцирование. Вычислительный граф, правило цепочки, forward-mode и reverse-mode autodiff. [Neural_Networks](DL/Neural_Networks.md)
5. Оптимизация в DL. Gradient descent, SGD, Momentum, Nesterov, RMSProp, Adam/AdamW, роль learning rate. [Neural_Networks](DL/Neural_Networks.md) [optimizers](DL/optimizers.md)
6. Стабилизация и регуляризация обучения. Инициализация, нормализация, dropout, weight decay, early stopping, data augmentation. [Neural_Networks](DL/Neural_Networks.md)
7. Практический PyTorch-пайплайн. Tensor, autograd, nn.Module, Dataset, DataLoader, train/eval. [PyTorch](DL/PyTorch.md)
8. Диагностика обучения нейросети. Переобучение, tiny-overfit test, анализ train/validation curves, leakage, подбор гиперпараметров. [Neural_Networks](DL/Neural_Networks.md)
9. Изображение как тензор и мотивация CNN. Почему полносвязная сеть плохо использует структуру изображения. [CNN](DL/CNN.md)
10. Операция свертки. Локальность, разделение весов, padding, stride, receptive field, dilation. [CNN](DL/CNN.md)
11. Базовая CNN для классификации изображений. Feature maps, conv, pooling, classifier head. [CNN](DL/CNN.md)
12. Виды сверток и эффективные операции. 1x1 convolution, separable/depthwise convolution, transposed convolution. [CNN](DL/CNN.md)
13. Классические CNN-архитектуры. LeNet, AlexNet, VGG, Inception: как развивались идеи глубины, сверток и feature extraction. [CNN](DL/CNN.md)
14. ResNet и современные CNN-backbone. Skip connections, residual block, bottleneck, идеи эффективных архитектур. [CNN](DL/CNN.md)
15. Transfer learning в компьютерном зрении. Feature extraction, fine-tuning, заморозка слоев, использование предобученных backbone. [Transfer_Learning](DL/Transfer_Learning.md)
16. Семантическая сегментация. Постановка задачи, отличие от классификации, FCN, encoder-decoder подход, superpixels. [Semantic_Segmentation](DL/Semantic_Segmentation.md)
17. U-Net и современные подходы к сегментации. Skip connections, upsampling, multiscale features, transformer/promptable segmentation как расширения. [Semantic_Segmentation](DL/Semantic_Segmentation.md)
18. Instance, panoptic segmentation и human pose estimation. Чем эти задачи отличаются от semantic segmentation, какие идеи используются. [Semantic_Segmentation](DL/Semantic_Segmentation.md)
19. Метрики и функции потерь в сегментации. Pixel-wise cross-entropy, IoU/mIoU, Dice/F1, особенности оценки качества. [Semantic_Segmentation](DL/Semantic_Segmentation.md)
20. Object detection. Постановка задачи, bounding boxes, confidence, IoU, отличие detection от classification и segmentation. [Object_Detection](DL/Object_Detection.md)
21. Метрики и post-processing в детекции. Precision/recall, AP/mAP, Non-Maximum Suppression. [Object_Detection](DL/Object_Detection.md)
22. Two-stage object detection. Sliding window, R-CNN, Fast R-CNN, Faster R-CNN, Region Proposal Network. [Object_Detection](DL/Object_Detection.md)
23. One-stage object detection. YOLO, SSD, RetinaNet, anchors, anchor-free идеи, trade-off скорости и качества. [Object_Detection](DL/Object_Detection.md)
24. Knowledge distillation. Teacher-student схема, soft labels, temperature, KL-divergence, зачем нужна дистилляция. [Knowledge_Distillation](DL/Knowledge_Distillation.md)
25. Варианты и применения дистилляции. Дистилляция логитов, признаков и attention, online/offline/self-distillation, dataset distillation. [Knowledge_Distillation](DL/Knowledge_Distillation.md)
26. Векторные представления слов. One-hot, distributional hypothesis, count-based embeddings, co-occurrence matrix, PMI/PPMI, LSA. [word2vec](DL/word2vec.md)
27. Word2Vec и GloVe. Skip-gram, CBOW, negative sampling, интерпретация embedding space, аналогии и косинусная близость. [word2vec](DL/word2vec.md)
28. Рекуррентные нейронные сети. Последовательные данные, hidden state, parameter sharing, BPTT, many-to-one/many-to-many задачи. [RNN](DL/RNN.md)
29. Проблемы RNN и gated-архитектуры. Vanishing/exploding gradients, LSTM, GRU, teacher forcing, scheduled sampling. [RNN](DL/RNN.md)
30. Seq2seq и attention до трансформеров. Encoder-decoder, bottleneck одного вектора, attention для машинного перевода. [RNN](DL/RNN.md)
31. Токенизация текста. Word-level, char-level, subword, BPE, WordPiece, byte-level BPE, Unicode/UTF-8. [Tokenization](DL/Tokenization.md)
32. Механизм внимания. Query, Key, Value, scaled dot-product attention, multi-head attention. [Transformer](DL/Transformer.md)
33. Архитектура Transformer. Self-attention, masked attention, cross-attention, FFN, residual connections, LayerNorm, positional encoding. [Transformer](DL/Transformer.md)
34. Pretraining в NLP. Зачем нужен self-supervised pretraining, encoder-only, encoder-decoder и decoder-only подходы. [Pretraining в NLP](DL/Pretraining_NLP.md)
35. BERT-подобные и T5-подобные модели. Masked language modeling, text-to-text постановка, fine-tuning под downstream tasks. [Pretraining в NLP](DL/Pretraining_NLP.md)
36. GPT-подобные модели. Autoregressive generation, next-token prediction, decoder-only архитектура, in-context learning. [Pretraining в NLP](DL/Pretraining_NLP.md)
37. Масштабирование трансформеров и MoE. Scaling laws, compute/data/model size, sparse experts, router, балансировка экспертов. [Transformer Scaling](DL/37-Transformer_Scaling)
38. Дообучение LLM. Full fine-tuning, supervised fine-tuning, instruction tuning, PEFT, adapters, LoRA, prefix/prompt tuning. [LLM fine-tuning](38-39-LLM_fine-tuning.md)
39. Preference tuning и RLHF. Reward model, человеческие предпочтения, отличие RLHF от SFT, роль KL-регуляризации. [LLM fine-tuning](38-39-LLM_fine-tuning.md)
40. In-context learning, prompting и Chain-of-Thought. Few-shot prompting, reasoning prompts, ограничения промптового подхода. [Prompting](DL/40-41-Prompting)
41. Catastrophic forgetting при дообучении. Почему возникает, чем опасен, какие общие способы смягчения используются. [Prompting](DL/40-41-Prompting)
42. Инференс LLM. Training vs inference, autoregressive generation, latency, KV-cache, почему генерация дорогая.
43. Ускорение и сжатие LLM на инференсе. Efficient attention, квантизация INT8/INT4, trade-off памяти, скорости и качества.
44. Декодирование LLM. Greedy decoding, sampling, temperature, top-k/top-p, beam search, связь стратегии декодирования с качеством ответа.
45. RAG, длинный контекст и агенты. Как добавлять внешнее знание без переобучения, retrieval, tools, ограничения агентных схем.
46. Звук как сигнал. Waveform, sampling, quantization, PCM, теорема Найквиста-Шеннона, aliasing.
47. Частотное представление звука. Fourier transform, DFT/FFT, STFT, спектрограмма, mel-scale, log-mel features, MFCC.
48. Automatic Speech Recognition. Постановка STT-задачи, уровни токенизации, WER/CER, alignment problem.
49. CTC для распознавания речи. Blank token, collapse operation, CTC loss, greedy/beam decoding.
50. Архитектуры ASR и TTS. RNN-T/LAS как альтернативы CTC; TTS pipeline: text normalization, acoustic model, vocoder.
51. Vision Transformer. Patch embedding, positional encoding, self-attention для изображений, сравнение ViT и CNN. [ViT](<DL/51-Vision Transformer. Patch embedding, positional encoding, self-attention для изображений, сравнение ViT и CNN.md>)
52. GAN. Генератор, дискриминатор, adversarial objective, min-max игра, основные проблемы обучения. [GAN](<DL/52-GAN. Генератор, дискриминатор, adversarial objective, min-max игра, основные проблемы обучения..md>)
53. Варианты и стабилизация GAN. Mode collapse, DCGAN, conditional GAN, pix2pix, CycleGAN, WGAN/WGAN-GP. [GAN variants](<DL/53-Варианты и стабилизация GAN. Mode collapse, DCGAN, conditional GAN, pix2pix, CycleGAN, WGAN, WGAN-GP.md>)
54. Autoencoders и VAE. Reconstruction, bottleneck, latent space, ELBO, KL-divergence, reparameterization trick. [VAE](<DL/54-Autoencoders и VAE. Reconstruction, bottleneck, latent space, ELBO, KL-divergence, reparameterization trick..md>)
55. Normalizing Flows и дискретные латентные модели. Обратимые преобразования, change of variables, exact likelihood, идея VQ-VAE. [Norm Flows](<DL/55-Normalizing Flows и дискретные латентные модели. Обратимые преобразования, change of variables, exact likelihood, идея VQ-VAE..md>)
56. Графы как данные и GNN. Node/edge/graph-level задачи, message passing, агрегация соседей, примеры GCN/GraphSAGE/GAT. [GNN](<DL/56-Графы как данные и GNN. Node-edge-graph-level задачи, message passing, агрегация соседей, примеры GCN-GraphSAGE-GAT..md>)
57. Проблемы и современные направления GNN. Oversmoothing, oversquashing, heterophily, масштабирование и graph foundation models. [GNN problems](<DL/57-Проблемы и современные направления GNN. Oversmoothing, oversquashing, heterophily, масштабирование и graph foundation models..md>)
58. Диффузионные модели. Генерация из шума, score matching, denoising diffusion, forward/reverse process, U-Net. [Diffusion](<DL/58-Диффузионные модели. Генерация из шума, score matching, denoising diffusion, forward-reverse process, U-Net..md>)
59. Условная и ускоренная диффузия. Guidance, text-to-image, DDIM/solvers/distillation, inpainting. [Conditional Diffusion](<DL/59-Условная и ускоренная диффузия. Guidance, text-to-image, DDIM,solvers, distillation, inpainting..md>)
60. Мультимодальные задачи и CLIP. Image-text retrieval, zero-shot classification, contrastive learning, общее пространство изображений и текста. [CLIP](<DL/60-Мультимодальные задачи и CLIP. Image-text retrieval, zero-shot classification, contrastive learning, общее пространство изображений и текста..md>)
61. Vision-Language Models и MLLM. Captioning, VQA, OCR, BLIP/LLaVA-подходы, visual encoder, projector, LLM, visual tokens. [VLM](<DL/61-Vision-Language Models и MLLM. Captioning, VQA, OCR, BLIP LLaVA-подходы, visual encoder, projector, LLM, visual tokens..md>)
62. Оценка и ограничения мультимодальных моделей. Бенчмарки, работа с большими изображениями, multi-image режим, типичные ошибки VLM. [VLM eval](<DL/62-Оценка и ограничения мультимодальных моделей. Бенчмарки, работа с большими изображениями, multi-image режим, типичные ошибки VLM..md>)

### Примечание к подчеркиванию

Подчеркнутые элементы указаны на уровне связи: ожидается понимание, что соответствующая сущность используется для данной задачи или идеи, без строгой постановки и формальных определений.

---

## Формат экзамена

Устный экзамен не предполагает заранее расписанные микробаллы за каждую строку ответа. Траектория опроса следующая: экзаменатор слушает ответ на билет, затем задает уточняющие и дополнительные вопросы, чтобы понять не только то, что студент выучил, но и насколько он понимает материал.

### Общие принципы оценивания

1. Проверяется понимание, а не заучивание текста. Студент должен уметь объяснять определения, формулы, алгоритмы и идеи своими словами, а также понимать, зачем они нужны в моделях и методах.

2. При ответе на билет тоже могут задаваться вопросы на глубину понимания. Полный ответ на билет включает не только перечисление известных фактов, но и объяснение того, как работают соответствующие методы, какие формулы в них используются, почему они выглядят именно так и при каких условиях применимы.

3. Дополнительные вопросы не являются произвольными вопросами «о чем угодно». Они задаются в рамках материала курса, заявленных пререквизитов и математических идей, которые используются в курсе: матстатистика, теория вероятности, линейная алгебра, математический анализ, методы оптимизации и машинное обучение.

4. Количество дополнительных вопросов заранее не фиксируется. Экзаменатор задает столько вопросов, сколько требуется, чтобы в пределах отведенного времени определить уровень владения материалом. Обычно вопросы идут от уточняющих и базовых к более глубоким.

5. Оценка ставится по совокупности ответов. Отсутствие ответа на один глубокий вопрос не означает низкую оценку, если базовый материал усвоен.

### Структура оценки

Итоговая оценка выставляется по 10-балльной шкале.

- Ответ на билет: до 4 баллов. Оцениваются корректность основных определений, теорем, алгоритмов и формул; понимание идеи метода; умение объяснить, как работает соответствующий метод, зачем используются его формулы и при каких условиях он применим.

- Дополнительные и уточняющие вопросы: до 6 баллов. Оцениваются глубина и ширина понимания материала курса: умение отвечать на уточнения, связывать темы между собой, объяснять математический смысл формул, применять идеи к новым, но связанным с курсом ситуациям, обсуждать ограничения методов и крайние случаи.

---

## Критерии уровней ответа

### Ответ на билет (до 4 баллов)

**3.5–4 балла:** Полный, точный и осмысленный ответ. Студент не только воспроизводит определения и формулы, но и объясняет их роль: что оптимизируется, какие распределения или величины входят в выражение, какие предположения используются, почему метод работает именно так и при каких условиях применим.

**2.5–3.4 балла:** Основной материал по билету изложен правильно, но есть отдельные неточности, неполные объяснения или слабые места в интерпретации формул. После уточняющих вопросов студент способен исправиться самостоятельно.

**1.5–2.4 балла:** Ответ частичный: базовые термины и отдельные факты названы, но понимание фрагментарное; студент с трудом объясняет смысл формул, связи с оптимизацией или статистической постановкой задачи.

**0–1.4 балла:** Существенные ошибки в базовых понятиях, отсутствие ответа по значимой части билета, невозможность объяснить основные формулы или алгоритм даже на уровне идеи.

### Дополнительные и уточняющие вопросы (до 6 баллов)

**5–6 баллов:** Студент уверенно отвечает на глубокие вопросы, строит связи между разделами курса, между материалом deep learning, матстатистикой и оптимизацией, ориентируется при измененных условиях постановки задачи и может объяснить ограничения методов.

**3.5–4.9 балла:** Студент хорошо отвечает на большинство уточняющих вопросов, понимает основные связи между темами, но может теряться на более глубоких вопросах, крайних случаях или при переносе идеи в новую постановку.

**2–3.4 балла:** Студент отвечает преимущественно на базовые дополнительные вопросы; понимание связей между темами ограничено; математическая интерпретация формул неполная.

**0–1.9 балла:** Студент не отвечает на простые уточнения по материалу курса, не может объяснить смысл ключевых формул, путает базовые понятия или демонстрирует отсутствие системного понимания.

---

## Что означает «понимать математику внутри нейросетей»

В рамках экзамена ожидается, что студент умеет связывать методы deep learning с математическими объектами, которые в них используются. Это не означает, что нужно наизусть воспроизводить все возможные доказательства или знать материал, не входящий в курс. Но если в курсе используется определенная функция потерь, вероятностная постановка, оптимизационный метод или дивергенция, студент должен понимать их роль.

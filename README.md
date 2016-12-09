## Auxiliary Deep Generative Models

Chainer implementation of the papers:

- [Auxiliary Deep Generative Models](http://arxiv.org/abs/1602.05473)
- [Improving Stochastic Gradient Descent with Feedback](https://arxiv.org/abs/1611.01505)
- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)
- [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)

[この記事](http://musyoku.github.io/2016/09/10/Auxiliary-Deep-Generative-Models/)で実装したコードです。

See also:
- [VAE](https://github.com/musyoku/variational-autoencoder)
- [AAE](https://github.com/musyoku/adversarial-autoencoder)

### Requirements

- Chainer 1.17
- Pylab
- pandas

this code contains following repos:

- [chainer-sequential](https://github.com/musyoku/chainer-sequential)

## Runnning

### ADGM

run `train_adgm/train.py`

### SDGM

run `train_sdgm/train.py`

## Validation Accuracy

![acuracy](http://musyoku.github.io/images/post/2016-09-10/adgm_graph.png)

## Analogies

### ADGM

![analogy](http://musyoku.github.io/images/post/2016-09-10/analogy_adgm.png)

### SDGM

![analogy](http://musyoku.github.io/images/post/2016-09-10/analogy_sdgm.png)
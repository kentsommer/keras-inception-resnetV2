# Keras Inception ResNet V2
Keras implementation of Google's inception-resnet-v2 model with (**coming soon**) ported weights!

As described in:
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi)](https://arxiv.org/abs/1602.07261)

Note this Keras implementation tries to follow the [tf.slim definition](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py) as closely as possible.

Pre-Trained weights (**once they are ported**) for this Keras model can be found here (ported from the tf.slim ckpt): https://github.com/kentsommer/keras-inception-resnetV2/releases

You can evaluate a sample image by performing the following (weights **once they are ported** are downloaded automatically):
* ```$ python evaluate_image.py```
```
Loaded Model Weights!
Class is: African elephant, Loxodonta africana
Certainty is: 0.868498
```

# Performance Metrics (@Top5, @Top1)

Error rate on non-blacklisted subset of ILSVRC2012 Validation Dataset (Single Crop):
* Top@1 Error: **coming soon**
* Top@5 Error: **coming soon**

Error rate listed in the paper on non-blacklisted subset of ILSVRC2012 Validation Dataset (Single Crop):
* Top@1 Error: 19.9%
* Top@5 Error: 4.9%

# News
2/8/2017:

1. The model has been added. I will work on porting the weights over the next few days

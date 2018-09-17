# Transfer Learning on Inception V3

I was curious what types of CNN people generally use for application. Some search indicates that [Keras](https://keras.io/applications/#inceptionv3) provides some most common award-winning CNN models like below

![Models](models.png)

I mainly this [tutorail](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2). In fact, a lot of code samples on [Keras official website](https://keras.io/applications/#inceptionv3) as well.

The code in image_recognition contains produces top 3 ImageNet labels of some existing model. I changed the code so that we can recognize several images at the same time.

If the thing you want to recognize is in the 1000 ImageNet labels, then no need to train the new model. However, often people may want to do finer-grain recognition, like flower for example, then you will want to do transfer learning.

In this case, I tried with this [flower dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) with around 4000 images of 5 different flower types.

Let's see how transfer learning plays a big role here.





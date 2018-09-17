import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

from keras.preprocessing import image
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
#from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

#model = ResNet50(weights='imagenet')
#model = VGG16(weights='imagenet')
#model = MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
#model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = InceptionV3(weights='imagenet')
# default arget size for Incpetion V3 is 299, default for others are 244
target_size = (299, 299)

def predict(model, img_names, target_size, top_n=3):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  xs = []
  for img_name in img_names:
      img = Image.open(img_name)
      if img.size != target_size:
        img = img.resize(target_size)
      x = image.img_to_array(img)
      xs.append(x)

  xs = np.asarray(xs)
  xs = preprocess_input(xs)
  preds = model.predict(xs)
  res = decode_predictions(preds, top=top_n)

  for i in range(len(img_names)):
      print img_names[i]
      print res[i]
      print 

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--folder", help="path to folder")
  args = a.parse_args()

  if args.image is None and args.folder is None:
    a.print_help()
    sys.exit(1)

  if args.image is not None:
    preds = predict(model, [args.image], target_size)

  if args.folder is not None:
    img_names = [args.folder + "/" + filename for filename in os.listdir(args.folder)]
    predict(model, img_names, target_size)

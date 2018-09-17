import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


target_size = (229, 229) #fixed size for InceptionV3 architecture

def find_label(l):
  m = {0: "daisy", 1: "dandelion", 2: "rose", 3: "sunflower", 4: "tulip"}
  highest = 0
  index = 0
  for i in range(5):
      if (l[i] > highest):
          highest = l[i]
          index = i
  return (m[index], highest)

def predict(model, img_names, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
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
  #x = np.expand_dims(x, axis=0)
  xs = preprocess_input(xs)
  preds = model.predict(xs)

  for i in range(len(img_names)):
      print "file: " + img_names[i]
      label, p = find_label(preds[i])
      print "prediction: " + label + " with probability " + str(p)
      print

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--folder", help="path to folder")
  a.add_argument("--model")
  args = a.parse_args()

  if args.image is None and args.folder is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    preds = predict(model, [args.image], target_size)

  if args.folder is not None:
    img_names = [args.folder + "/" + filename for filename in os.listdir(args.folder)]
    predict(model, img_names, target_size)

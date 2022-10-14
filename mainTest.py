import cv2
from keras.models import load_model
from PIL import Image
from matplotlib.pyplot import axis
import numpy as np
model=load_model('Dementia20Epochs128.h5')

image = cv2.imread('E:\\Deep Learning Project\\Dementia Image Classification\\pred\\pred04y.jpg')

img=Image.fromarray(image)

img=img.resize((128,128))

img=np.array(img)
input_img=np.expand_dims(img, axis=0)

result=model.predict_classes(input_img)
print(result)
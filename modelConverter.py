from keras.models import load_model
import tensorflow as tf 

model=load_model('Dementia50Epochs128early.h5')

converter=tf.lite.TFLiteConverter.from_keras_model(model)

tf_lite_model=converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tf_lite_model)
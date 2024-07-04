import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from  PIL import Image,ImageOps

def load_model():
  model=tf.keras.models.load_model('/content/my_model.h5')
  return model

with st.spinner('loading...'):
  model=load_model()

def predict(image_data,model):
  size=(28,28)
  image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
  image=np.asarray(image)
  img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  img_reshape=img.reshape((1,28,28,1))
  prediction=model.predict(img_reshape)
  return prediction

st.title('Handwritten Digit Recognition')
files=st.file_uploader('',type=['jpg','png'])

if files is None:
  st.text('Please upload an image file')
else:
  image=Image.open(files)
  st.image(image,use_column_width=True)
  predictions=predict(image,model)
  score=tf.nn.softmax(predictions[0])

  st.write(predictions)
  st.write(score)

  class_names=['0','1','2','3','4','5','6','7','8','9']
  st.write("This is most likey belongs to {} with a {:.2f} percent confidence."
  .format(class_names[np.argmax(score)],100*np.max(score)))

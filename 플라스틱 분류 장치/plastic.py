import time
time_start = time.time()
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
import keras.applications.mobilenet_v2
import tensorflow as tf
import numpy as np
import cv2
import json
import requests
from io import BytesIO
from pathlib import Path
import shutil
time_import = time.time()

#Flask for background running
app = Flask(__name__, template_folder=os.path.abspath("../"))

#model & label define
global model
class_names=['PP','PET','PS','HDPE','LDPE']
num_class = len(class_names)

model = tf.keras.Sequential([
tf.keras.applications.MobileNetV2(weights='imagenet',input_shape=(224,224,3),pooling=None,classes=1000),
#tf.keras.layers.Dense(50,activation='relu'),
tf.keras.layers.Dense(num_class,activation='softmax',activity_regularizer = tf.keras.regularizers.l1(0.04))
])

#import pre-trained weight
time_define_model = time.time()
checkpoint_path ="static/weight/cp.ckpt"
model.load_weights(checkpoint_path)
time_load_weights = time.time()

#move image url to destination and return its path
def Move(url):
    NAME= os.path.basename(url)
    destination = "static/destination/"+NAME
    if os.path.exists(url):
        shutil.move(url,destination)
    return destination

#지금은 분류를 재분류 하는 형태인데 나중에 대대적인 수정이 필요할듯!
def determine(true_label):
  LAST_label=int(true_label)%5
  if LAST_label==0:
    LAST_label='PP'
  if LAST_label==1:
    LAST_label='PET'
  if LAST_label==2:
    LAST_label='PS'
  if LAST_label==3:
    LAST_label='HDPE'
  if LAST_label==4:
    LAST_label='LDPE' 
  return LAST_label

#when only click analysis snapshot of image accumulate in destination so clean it
def clean():
  for f in Path('static/destination').glob('*.jpg'):
    try:
        f.unlink()
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

#predict
def predict(path):
    img = cv2.imread(path)
    xy = cv2.resize(img,(224,224))/255
    xy = np.expand_dims(xy, axis=0)
    preds = model.predict(xy)
    print(class_names[np.argmax(preds)],np.max(preds))
    print(np.argmax(preds))
    return determine(np.argmax(preds))


global Class
Class = None
global pred_class
pred_class=None
@app.route('/capture',methods=['GET','POST'])
def capture():
    if request.method == "GET":
        return render_template("gui/plastic.html")

    if request.method == "POST":
        url = request.form["url"]
        functionType = request.form["functionType"]

        global pred_class
        start=time.time()
        A=predict(url)
        print(time.time()-start)
        pred_class=A
        clean()
        destination = Move(url)
        
        return jsonify(
            preds=A,
            img=url,
            destination=destination
        )



@app.route("/save", methods=["GET", "POST"])
def save_img():
    if request.method=="POST":
        try:
            global Class
            Class = request.form["checkClass"]
            print(Class)

        except:
            print("except")
    if pred_class!=None:
        file_name_with_ext = request.form['fileName']
        #file_name_with_ext = os.path.basename()
        file_name = os.path.splitext(file_name_with_ext)[0]
        file_path = 'static/destination/'+ file_name_with_ext
        #file_path = '/media/kimm/UNTITLED1/engine/static/destination/'+ file_name_with_ext
    
        if Class == pred_class:
            path ="/home/kimm/Desktop/engine/static/correct/"+Class+"/"
            #path ="/media/kimm/UNTITLED/static/correct/"+Class+"/"
            #path ="static/correct/"+Class+"/"
        else: 
            #print(pred_class)
            #path ="static/incorrect/"+Class+"/"
            path ="/home/kimm/Desktop/engine/static/incorrect/"+Class+"/"
            
            #path ="/media/kimm/UNTITLED/static/incorrect/"+Class+"/"

    final = path+file_name+"_"+pred_class+"_"+Class+'.jpg'
    shutil.copy(file_path,final)

    return jsonify(
            result="결과 저장"
    )
 
time_end = time.time()    
print('-'*50)
print('import packages : ', round(time_import-time_start,2), 'seconds')
print('-'*50)
print('define models   : ', round(time_define_model-time_import,2), 'seconds')
print('-'*50)
print('load weights    : ', round(time_load_weights-time_define_model,2), 'seconds')
print('-'*50)
print('total time       : ', round(time_end-time_start,2), 'seconds')
print('-'*50)
app.run(threaded=True)

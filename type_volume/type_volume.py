# -*- coding: utf-8 -*-
# # To execute:
# #         python pc.py --config config/pc.conf
# # 
# # parameters on the cctv is stored in pc.conf
# # prameters on the database is stored in db.conf


# import the necessary packages
from utils.type_volume_app import PCApp
from imutils.video import FileVideoStream
import sys
import argparse
import json
from datetime import datetime,timedelta
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import time
import os



# +
def main(model):
	# construct the argument parse and parse the arguments
#	print("main진입")    
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--config", type=str,
		required=True, help="path to cctv config file")
	args = vars(ap.parse_args())
#	print("main진행")  
    
	with open(args['config']) as json_file:
		data = json.load(json_file)
		pc_info = data

	which_cctv = 0

    # load PCApp class
	app = PCApp(pc_info, which_cctv,model)
#	print("app진행")  
#	app = PCApp(model,model1)    

# load detector python module
# 	app.load_detector()    
# 	app.load_tracker()
# 	print("[INFO]  opening video stream...")
	#vs = FileVideoStream(pc_info['cctv'][which_cctv]['address']).start()
	app.video_loop()
#	print("app_loop진행")    

# +
#start = datetime.now()
#now = datetime.now()
#d = now-start
#####path = "./volume_model"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#####volume_model = load_model(path)
#####volume_model.summary()
model_1 = load_model('./volume_model')
model_1.summary()
print("load model success")

start = datetime.now()+timedelta(hours=9)
now = datetime.now()+timedelta(hours=9)
d = now-start
#volume_model = 1
# print(2)
#####path = "./type_model"
# print(3)
# type_model = tf.keras.models.load_model(path)
#####type_model = 2
#print(4)

if __name__ == '__main__':    
	while True:
		now = datetime.now()+timedelta(hours=9)
		a = d.seconds
		d = now-start
		b = d.seconds
		model = model_1
#		print(b)
		if b%600 ==0:
			print(b)
#			time.sleep(3)
			main(model) #main(volume_model, type_model)
			print('*****done*****')
#			break

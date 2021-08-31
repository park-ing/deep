# -*- coding: utf-8 -*-
import cv2
import imutils
from matplotlib import image
from imutils.video import FPS
from imutils.video import FileVideoStream
import os
#from utils.centroidtracker import CentroidTracker
#from utils.trackableobject import TrackableObject
from datetime import datetime, timedelta
from tensorflow import keras
import tensorflow as tf
import numpy as np
import dlib
import json
import random
from random import gauss
from queue import Queue
from scipy import ndimage
from utils.type_volume_utils import *
import time
import os 
from keras.models import load_model
from PIL import Image


# +
class PCApp:
	def __init__(self,pc_info,which_cctv,model):    
		self.cctv_dont_show         = pc_info['cctv'][which_cctv]['dont_show']
		self.cctv_resize_width      = pc_info['cctv'][which_cctv]['resize_width']
		self.cctv_resize_width_show = pc_info['cctv'][which_cctv]['resize_width_show']
		self.cctv_roi_ratio         = pc_info['cctv'][which_cctv]['roi_ratio']
		self.cctv_use_roi           = pc_info['cctv'][which_cctv]['use_roi']
		self.cctv_output_path       = pc_info['cctv'][which_cctv]['output_path']
		self.cctv_address           = pc_info['cctv'][which_cctv]['address']
		self.c_cctv_info_id         = pc_info['cctv'][which_cctv]['c_cctv_info_id']
		self.cctv_typy              = pc_info['cctv'][which_cctv]['cctv_type']
		self.ES_urls                = pc_info['ES'][which_cctv]['urls']
		self.ES_index               = pc_info['ES'][which_cctv]['index']
		self.ES_use_yn              = pc_info['ES'][which_cctv]['use_yn']   
		self.model  = model       

	def video_loop(self):
		vs = FileVideoStream(self.cctv_address).start()
		es = Elasticsearch(self.ES_urls, timeout=30, max_retries=10, retry_on_timeout=True)
		self.ES_es = es
		totalFrames = 0
		W = None
		H = None
		q = Queue(maxsize = 5)
# 		start = datetime.now()
# 		now = datetime.now()
# 		d = now-start
		fps = FPS().start()
# 		while True:
		for i in range(1):
			frame = vs.read()
			checkPath = "{}/{}".format(self.c_cctv_info_id,(datetime.now()+timedelta(hours=9)).strftime('%Y%m%d'))
			dirPath = os.path.sep.join((self.cctv_output_path, checkPath))            
			try:
				if not os.path.isdir(dirPath):
					os.mkdir(dirPath)
			except:
				pass  
			try:
				frame = imutils.resize(frame, width = self.cctv_resize_width) 
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				if self.cctv_use_roi:
						roi_frame, output = img_overlay(frame, 0.5, self.cctv_roi_ratio)
						frame = roi_frame
				if W is None or H is None:
					(H, W) = frame.shape[:2]
				rects = []
# 				now = datetime.now()
# 				a = d.seconds
# 				d = now-start
# 				b = d.seconds
# 				if a!=b:
# 					if b%300 ==0:
# 						print(b,'seconds passed')
# 						filename = "{}_{}.jpg".format(self.c_cctv_info_id, datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
# 						save_counted_to_image(self, frame, 'STAY', dirPath, filename)
# 				print(b,'seconds passed')
				filename = "{}_{}.jpg".format(self.c_cctv_info_id, (datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%dT%H:%M:%S'))
				save_counted_to_image(self, frame, 'STAY', dirPath, filename)
# 				if self.ES_use_yn:
# 					push_data_ES(self, int(0), int(1), int(2), int(3), int(4), int(5), int(0), int(0), int(1), int(1), int(2), int(2),[0.5,0.1,0.1,0.1,0.1,0.1],[0.1,0.5,0.1,0.1,0.1,0.1],[0.1,0.1,0.5,0.1,0.1,0.1],[0.1,0.1,0.1,0.5,0.1,0.1],[0.1,0.1,0.1,0.1,0.5,0.1],[0.1,0.1,0.1,0.1,0.1,0.5],[0.8,0.1,0.1],[0.8,0.1,0.1],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],dirPath, filename)
				if not self.cctv_dont_show:
					cv2.imshow("PC App", imutils.resize(frame, width=self.cctv_resize_width_show))
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"):
						break
				totalFrames += 1
				fps.update()
			except Exception as e:
				print("except")
				print(e)
				totalFrames +=1
				vs.stop()
				del vs
				vs = FileVideoStream(self.cctv_address).start()
		fps.stop()
		vs.stop()
        ##텐서플로우 default 메모리 과다 할당 방지

		try:
#			config = tf.compat.v1.ConfigProto()
#			config.gpu_options.allow_growth = True
#			session = tf.compat.v1.Session(config=config)
        ##
			pil_image = Image.fromarray(frame)
			print(pil_image.size)
			img = pil_image.convert('RGB')
			img = img.resize((224,224))
			print(img.size)
        
			img_array = []
			img_array.append(np.asarray(img))
			images = np.array(img_array)
			images = images.astype(np.float32) /225.
			print(images.shape) #(1,224,224,3)

			print(self.c_cctv_info_id) 
			ind1 = []      

			if self.c_cctv_info_id == 7001: # 캔/철
				type_num = 0
				ind1 = [0.51,0.04,0.08,0.12,0.23,0.02]
			elif self.c_cctv_info_id == 7002: # 일반플라스틱
				type_num = 1
				ind1 = [0.04,0.51,0.08,0.12,0.23,0.02]
			elif self.c_cctv_info_id == 7003: # 투명플라스틱
				type_num = 2
				ind1 = [0.08,0.04,0.51,0.12,0.23,0.02]                
			elif self.c_cctv_info_id == 7004: # 비닐(1)
				type_num = 3
				ind1 = [0.12,0.04,0.08,0.51,0.23,0.02]
			elif self.c_cctv_info_id == 7005: # 비닐(2)
				type_num = 4
				ind1 = [0.23,0.04,0.08,0.12,0.51,0.02]
			elif self.c_cctv_info_id == 7006: # 스티로폼
				type_num = 5
				ind1 = [0.02,0.04,0.08,0.12,0.23,0.51]
                
			print(type_num,ind1)
			print(np.argmax(ind1))        
##			model_1 = load_model('./volume_model')
#			model_1.summary()
##			print("load model success")
			Model = self.model
			model_list = Model.predict(images)
			print("------------------------------")
#				print(model_list)
#				print('[INFO]  elapsed time: {:.2f}'.format(fps.elapsed()))
#				print('[INFO]  approx. FPS: {:.2f}'.format(fps.fps()))
			print("check")      
        
			if self.ES_use_yn:
				print("type :",np.argmax(ind1))
				print("v0 :",model_list[0][0])
				print("v1 :",model_list[0][1])
				print("v2 :",model_list[0][2])
				print("pred :",np.argmax(model_list))
#				push_data_ES(self, 1,i, np.argmax(ind1[i]),ind1[i], np.argmax(ind[i]),ind[i],1)
				push_data_ES(self, 1,type_num, np.argmax(ind1),ind1, np.argmax(model_list),model_list,1)
				print('push done')
				print("===========================")
		except Exception as e:
			print("except")
			print(e)


# 사진 찍어서 보관하고 + 예측값 전달 (같은 시간에 둘다..? / 아님 두개 시간 따로??)
# -



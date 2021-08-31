import cv2
import imutils
#from imutils.video import FPS
import os
#from utils.centroidtracker import CentroidTracker
#from utils.trackableobject import TrackableObject
#from utils.dbtools import DBtools
import numpy as np
import dlib
#import json

#from queue import Queue
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import certifi
from elasticsearch.helpers import bulk

import paramiko
from scp import SCPClient, SCPException


class SSHManager:
    """
    usage:
        >>> import SSHManager
        >>> ssh_manager = SSHManager()
        >>> ssh_manager.create_ssh_client(hostname, username, password)
        >>> ssh_manager.send_command("ls -al")
        >>> ssh_manager.send_file("/path/to/local_path", "/path/to/remote_path")
        >>> ssh_manager.get_file("/path/to/remote_path", "/path/to/local_path")
        ...
        >>> ssh_manager.close_ssh_client()
    """
    def __init__(self):
        self.ssh_client = None


    def create_ssh_client(self, hostname, username, password,port):
        """Create SSH client session to remote server"""
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(hostname, username=username, password=password,port=port)
        else:
            print("SSH client session exist.")
            

    def close_ssh_client(self):
        """Close SSH client session"""
        self.ssh_client.close()

    def send_file(self, local_path, remote_path):
        """Send a single file to remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path)
        except SCPException as e:
#             raise SCPException
            print(e)

    def get_file(self, remote_path, local_path):
        """Get a single file from remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
        except SCPException as e:
#             raise SCPException
            print (e)

    def send_command(self, command):
        """Send a single command"""
        stdin, stdout, stderr = self.ssh_client.exec_command(command)
        return stdout.readlines()


def img_overlay(frame, alpha, roi_ratio):
    overlay = frame.copy()
    output = frame.copy()
    colors = [(186,226,239), (227,203,197)]

    (H, W) = frame.shape[:2]

    roi_coordinates ={
        'x1':int(roi_ratio[0]*W),
        'y1':int(roi_ratio[1]*H),
        'x2':int(roi_ratio[2]*W),
        'y2':int(roi_ratio[3]*H)
        }

    roi_x1 = roi_coordinates['x1']
    roi_y1 = roi_coordinates['y1']
    roi_x2 = roi_coordinates['x2']
    roi_y2 = roi_coordinates['y2']
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2, :]

    cv2.rectangle(overlay, (roi_x1,roi_y1), (roi_x2,roi_y2), (0,255,255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return roi_frame, output


def push_data_ES(self,cleanhouse_id, bin_id, Type, confidence_Type, Volume, confidence_Volume, status_id):
   day = ['mon','thu','wed','thr','fri','sat','sum']
   _data = {
       "ymdt_kst": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
       "day_of_week": (datetime.now()+timedelta(hours =9)).weekday()+1,
       "cleanhouse_id": cleanhouse_id,
       "bin_id": bin_id,
       "type_id" : Type,
       "volume_id" : Volume,
       "type_confidence" : confidence_Type,
       "volume_confidence" : confidence_Volume,
       "status_id":status_id
#        "img_file_path": c_img_file_path, 
#        "img_file_name":c_img_file_name
      }
   print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'))
#    print((datetime.now()+timedelta(hours =9)).weekday()+1)
   res = self.ES_es.index(index=self.ES_index, body = _data)


def push_data_ES_0(self, Type1, Type2, Type3, Type4, Type5, Type6, Volume1, Volume2, Volume3, Volume4, Volume5, Volume6, confidence_Type1, confidence_Type2, confidence_Type3, confidence_Type4, confidence_Type5, confidence_Type6, confidence_Volume1, confidence_Volume2, confidence_Volume3, confidence_Volume4, confidence_Volume5, confidence_Volume6):

   _data = {
       "ymdt_kst": datetime.now().strftime('%Y_%m_%d_T%H:%M:%S.%f'),
#        "cleanhouse_id": cleanhouse_id
       "Type_1" : Type1,
       "Type_2" : Type2,
       "Type_3" : Type3,
       "Type_4" : Type4,
       "Type_5" : Type5,
       "Type_6" : Type6,
       "Volume_1" : Volume1,
       "Volume_2" : Volume2,
       "Volume_3" : Volume3,
       "Volume_4" : Volume4,
       "Volume_5" : Volume5,
       "Volume_6" : Volume6,
       "confidence_Type_1" : confidence_Type1,
       "confidence_Type_2" : confidence_Type2,
       "confidence_Type_3" : confidence_Type3,
       "confidence_Type_4" : confidence_Type4,
       "confidence_Type_5" : confidence_Type5,
       "confidence_Type_6" : confidence_Type6,
       "confidence_Volume_1" : confidence_Volume1,
       "confidence_Volume_2" : confidence_Volume2,
       "confidence_Volume_3" : confidence_Volume3,
       "confidence_Volume_4" : confidence_Volume4,
       "confidence_Volume_5" : confidence_Volume5,
       "confidence_Volume_6" : confidence_Volume6,
#        "img_file_path": c_img_file_path, 
#        "img_file_name":c_img_file_name
      }

   res = self.ES_es.index(index=self.ES_index, body = _data)


def push_data_ES_old(self, c_type_info_id, c_object_type_info_id, c_object_x, c_object_y, c_object_vec_x, c_object_vec_y, c_section_info_id,  c_cctv_info_id, c_to_id, c_number_of_objects, c_direction_type_info_id, c_object_confidence, c_img_file_path, c_img_file_name):

   _data = {
       "ymdt_utc": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f'),
       "ymdt_kst": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
       "c_type_info_id": c_type_info_id,
       "c_object_type_info_id": c_object_type_info_id,
       "c_object_x": c_object_x,
       "c_object_y": c_object_y,
       "c_object_vec_x": c_object_vec_x,
       "c_object_vec_y": c_object_vec_y,
       "c_section_info_id": c_section_info_id,
       "c_cctv_info_id": c_cctv_info_id,
       "c_to_id": c_to_id,
       "c_number_of_objects": c_number_of_objects,
       "c_direction_type_info_id": c_direction_type_info_id,
       "c_object_confidence": c_object_confidence,
       "c_img_file_path":c_img_file_path,
       "c_img_file_name":c_img_file_name
      }

   res = self.ES_es.index(index=self.ES_index, body = _data)

def make_mn_size(IMG,m,n):
	imagess=[]
	for i in range(len(IMG)):
		img=IMG[i]
		a = cv2.resize(np.float32(img),(m,n), interpolation=cv2.INTER_AREA)
		imagess.append(a)
	return np.asarray(imagess)/255.


def get_data_ES(self, c_type_info_id, c_object_type_info_id, c_object_x, c_object_y, c_object_vec_x, c_object_vec_y, c_section_info_id,  c_cctv_info_id, c_to_id, c_number_of_objects, c_direction_type_info_id, c_object_confidence):
   _data = {
       "_index": self.ES_index,
       "ymdt_utc": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f'),
       "ymdt_kst": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
       "c_type_info_id": c_type_info_id,
       "c_object_type_info_id": c_object_type_info_id,
       "c_object_x": c_object_x,
       "c_object_y": c_object_y,
       "c_object_vec_x": c_object_vec_x,
       "c_object_vec_y": c_object_vec_y,
       "c_section_info_id": c_section_info_id,
       "c_cctv_info_id": c_cctv_info_id,
       "c_to_id": c_to_id,
       "c_number_of_objects": c_number_of_objects,
       "c_direction_type_info_id": c_direction_type_info_id,
       "c_object_confidence": c_object_confidence
       }

   return _data

def get_generator(_bulk):
        for _data in _bulk:
                yield _data


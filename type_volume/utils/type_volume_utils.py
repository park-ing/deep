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


# +
#def _save_counted_to_image(self, frame, centroid, text):
def save_counted_to_image(self, frame, text, dirPath, filename):
    _path = os.path.sep.join((dirPath, filename))
    cv2.imwrite(_path, frame.copy())

    file = '/home/dev/picture/'+filename
#     print(file)
    ssh_manager = SSHManager()
    ssh_manager.create_ssh_client("server-a.aigenn.com", "dev", "dev0319@",2221)
    
#     ssh_manager.send_file(_path, filename)
    ssh_manager.send_file(_path,file)

#     print(3)
    ssh_manager.close_ssh_client()
#     print(4)


# -

def gradient_line_x(pt1,pt2,x):
    pt1_x=pt1[0]
    pt1_y=pt1[1]

    pt2_x=pt2[0]
    pt2_y=pt2[1]

    if (pt2_x-pt1_x) !=0:
        gradient = (pt2_y-pt1_y)/(pt2_x-pt1_x)
        value = gradient*(x-pt1_x)+pt1_y
    else:
        value = pt1_x

    return value 

def gradient_line_y(pt1,pt2,x):
    pt1_x=pt1[1]
    pt1_y=pt1[0]

    pt2_x=pt2[1]
    pt2_y=pt2[0]

    if (pt2_x-pt1_x) !=0:
        gradient = (pt2_y-pt1_y)/(pt2_x-pt1_x)
        value = gradient*(x-pt1_x)+pt1_y
    else:
        value = pt1_x

    return value 

def draw_counting_areas_y(frame, counting_areas_y,gradient_y):
    (H, W) = frame.shape[:2]
    _line_position = counting_areas_y[0]['line_position']
    _offset = int(counting_areas_y[0]['offset']*H)
    _line = int(H*_line_position)
    cv2.line(frame, (0, _line + _offset), (W, int(-W*gradient_y +_line + _offset)), (227,203,197), 1)
    cv2.line(frame, (0, _line - _offset), (W, int(-W*gradient_y + _line - _offset)), (227,203,197), 1)
    y_under_start = (0, _line + _offset)
    y_under_end = (W, int(-W*gradient_y +_line + _offset))
    y_over_start = (0, _line - _offset)
    y_over_end = (W, int(-W*gradient_y + _line - _offset))
    return _line, _offset , y_under_start,y_under_end,y_over_start,y_over_end

def draw_counting_areas_x(frame, counting_areas_x,gradient_x):
    (H, W) = frame.shape[:2]
    _line_position = counting_areas_x[0]['line_position']
    _offset = int(counting_areas_x[0]['offset']*W)		
    _line = int(W*_line_position)
    cv2.line(frame, (_line + _offset, 0), (int(-W*gradient_x + _line + _offset), H), (186,226,239), 1)
    cv2.line(frame, (_line - _offset, 0), (int(-W*gradient_x+ _line - _offset), H), (186,226,239), 1)
    x_left_start =(_line - _offset, 0)
    x_left_end = (int(-W*gradient_x+ _line - _offset), H)
    x_right_start =(_line + _offset, 0)
    x_right_end = (int(-W*gradient_x + _line + _offset), H)
    return _line, _offset,x_left_start,x_left_end,x_right_start,x_right_end

def draw_direction_arrows1(frame, centroid, direction_x, direction_y):
    thickness = 2
    colors = [(255,255,0), (255,255,0)] # [color_x, color_y]
    line_type = cv2.LINE_AA 
    shift = 0
    tipLength = 0.2
    if direction_x < 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0]+int(direction_x), 
            centroid[1]), colors[0], thickness, line_type, shift, tipLength)
    elif direction_x > 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0]+int(direction_x), 
            centroid[1]), colors[0], thickness, line_type, shift, tipLength)

    if direction_y < 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0], centroid[1]+int(direction_y)), 
            colors[1], thickness, line_type, shift, tipLength) 
    elif direction_y > 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0] , centroid[1]+int(direction_y)), 
            colors[1], thickness, line_type, shift, tipLength)

def draw_direction_arrows(frame, centroid, direction_x, direction_y):
    thickness = 1
    colors = [(34,153,215), (161,86,64)] # [color_x, color_y]
    line_type = cv2.LINE_AA 
    shift = 0
    tipLength = 0.2
    if direction_x < 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0]+int(direction_x), 
            centroid[1]), colors[0], thickness, line_type, shift, tipLength)
    elif direction_x > 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0]+int(direction_x), 
            centroid[1]), colors[0], thickness, line_type, shift, tipLength)

    if direction_y < 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0], centroid[1]+int(direction_y)), 
            colors[1], thickness, line_type, shift, tipLength) 
    elif direction_y > 0:
    	cv2.arrowedLine(frame, (centroid[0] , centroid[1]), (centroid[0] , centroid[1]+int(direction_y)), 
            colors[1], thickness, line_type, shift, tipLength)

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

def draw_to_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
#    color = (128,128,128)
#    thickness = 1
#    r = 1
#    d = 2
 
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_counted_to_border(img, centroid, color):
#    x1,y1 = pt1
#    x2,y2 = pt2
#    offset = 15
    offset = 10
    x1,y1 = centroid[0]-offset, centroid[1]-offset 
    x2,y2 = centroid[0]+offset, centroid[1]+offset
#    color = (255,255,255)
    thickness = 1
    r = 5
    d = 5
 
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int"), pick

def put_counting_number_circle(frame, boxes):
	(H, W) = frame.shape[:2]
	cv2.circle(frame, (int(W*0.9), int(H*0.1)), int(W*0.05), (34,153,215), -1)
	cv2.putText(frame, "{}".format(len(boxes)), (int(W*0.9), int(H*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (161,86,64), int(W*0.005))

def put_counting_number_rectangle(img, number, centerX, centerY, color, thickness, r, d):
    (H, W) = img.shape[:2]
    offset = int(W*0.05)
    x1,y1 = centerX - offset, centerY - offset 
    x2,y2 = centerX + offset, centerY + offset 
    color = (196,196,196)
    color = (50, 50, 50)
    thickness = 5
    r = int(W*0.01)
    d = int(W*0.05)

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.putText(img, "{}".format(number), (centerX-int(offset/2), centerY+int(offset/2)), cv2.FONT_HERSHEY_SIMPLEX, W*0.0025, color, int(W*0.005))

def element_with_the_largest_frequency(queuelist):
    res = max(set(queuelist), key = queuelist.count)
    return res


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


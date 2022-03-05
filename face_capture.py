import sys
import numpy as np
import cv2

'''
# 녹색 배경 동영상
cap1 = cv2.VideoCapture('woman.mp4')

if not cap1.isOpened():
    print('video open failed!')
    sys.exit()

# 비오는 배경 동영상
cap2 = cv2.VideoCapture('raining.mp4')

if not cap2.isOpened():
    print('video open failed!')
    sys.exit()

# 두 동영상의 크기, FPS는 같다고 가정
frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame_cnt1:', frame_cnt1)
print('frame_cnt2:', frame_cnt2)

fps = cap1.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# 합성 여부 플래그
do_composit = False

# 전체 동영상 재생
while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        break
    
    # do_composit 플래그가 True일 때에만 합성
    if do_composit:
        ret2, frame2 = cap2.read()

        if not ret2:
            break

        # HSV 색 공간에서 녹색 영역을 검출하여 합성
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))
        cv2.copyTo(frame2, mask, frame1)

    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay)

    # 스페이스바를 누르면 do_composit 플래그를 변경
    if key == ord(' '):
        do_composit = not do_composit
    elif key == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
'''

import os
import sys
import glob
import cv2


def save_face(frame, p1, p2, filename):
    cp = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)

    w = p2[0] - p1[0]
    h = p2[1] - p1[1]

    if h * 3 > w * 4:
        w = round(h * 3 / 4)
    else:
        h = round(w * 4 / 3)

    x1 = cp[0] - w // 2
    y1 = cp[1] - h // 2
    if x1 < 0 and y1 < 0:
        return
    if x1 + w >= frame.shape[1] or y1 + h >= frame.shape[0]:
        return

    crop = frame[y1:y1+h, x1:x1+w]
    crop = cv2.resize(crop, dsize=(150, 200), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename, crop)


# Video Capture
cap = cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

# Network

model = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'opencv_face_detector/deploy.prototxt'

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# Output Directory & File Index

outdir = 'output'
prefix = outdir + '/face_'
file_idx = 1

try:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
except OSError:
    print('output folter create failed!')

png_list = glob.glob(prefix + '*.png')
if len(png_list) > 0:
    png_list.sort()
    last_file = png_list[-1]
    file_idx = int(last_file[-8:-4]) + 1

# Read Frames

cnt = 0
while True:
    _, frame = cap.read()
    if frame is None:
        break

	# Face Detection

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.8:
            break

		# Face found!

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        # Save face image as a png file

        cnt += 1

        if cnt % 10 == 0:
            filename = '{0}{1:04d}.png'.format(prefix, file_idx)
            save_face(frame, (x1, y1), (x2, y2), filename)
            file_idx += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(25) ##동영상이 너무 빨리 재생될 떄

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

import onnxruntime as rt
import os

session = rt.InferenceSession('study/src/sequential_opset10.onnx')

from PIL import Image
import numpy as np
fpaths = 'study/src/pet.jpg'
m = 224
n = 224
label = 'PET'
img_array = []
lab_array = []

# print(fname)  
img = Image.open(fpaths).convert('RGB')
print(img.size)
img = img.resize((m, n))
print(img.size)
img_array.append(np.asarray(img))
#print(img_array.shape)
lab_array.append(label)

images = np.array(img_array)
labels = np.array(lab_array)

images = images.astype(np.float32) /255.


input_name = session.get_inputs()[0].name
print(input_name)
pred_onnx = session.run(None, {input_name: images})[0]
print(pred_onnx)

print(rt.get_device())
print(np.argmax(pred_onnx))

print()

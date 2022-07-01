import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
for filepath in tqdm(os.listdir('images')):
  print(filepath)
  img = cv2.imread('images/'+filepath)
  frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  frame_threshold = 255-cv2.inRange(frame_HSV, (0, 0, 0), (255, 255, 120))
  frame_threshold = np.expand_dims(frame_threshold,axis=2)
  frame_threshold = np.concatenate([frame_threshold,frame_threshold,frame_threshold],axis=2)
  im = np.concatenate([img,frame_threshold],axis=1)
  cv2.imwrite('proc_images/'+filepath,frame_threshold)
  plt.figure(figsize=(20,10))
  plt.imshow(im)
  plt.show()

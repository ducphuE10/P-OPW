import cv2
import os

def extract(path_video, time):

  cap = cv2.VideoCapture(path_video)
  print(cap)
  i = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      if i%time == 0:
        # save_dir = '/content/drive/MyDrive/face_detect/'+path_video[29:-4]
        # if not os.path.isdir(save_dir):
        #   os.makedirs(save_dir)
        cv2.imwrite('img/img2/image'+str(i//time)+'.png',frame)
        # print('/image'+str(i//time)+'.png')

      i+=1
    else:
      break

extract('a16_s02_e01_rgb.avi', 12)
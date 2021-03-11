import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('test_images/Bebop_1.mp4')

# Check if camera opened successfully
if not cap.isOpened:
    print("Error opening video stream or file")

while cap.isOpened():

    ret, frame = cap.read()
    x = np.transpose(frame)
    x = [torch.as_tensor(x)]
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = model(x)
    if ret:
        cv2.imshow('Frame', frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


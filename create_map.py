import numpy as np
import cv2

# 1 is free, 0 is wall

a = np.array([
[1, 1, 1],
[1, 0, 1],
[1, 1, 1],
], float)
img_name = 'maps/test_3x3.jpg'

cv2.imshow('New map: ' + img_name, cv2.resize(a, (700, 700), interpolation=cv2.INTER_AREA))
cv2.waitKey()

a = np.array(a, dtype=np.uint8)*255
cv2.imwrite(img_name, a)

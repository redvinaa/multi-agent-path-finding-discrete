import numpy as np
import cv2

a = np.array([
[0, 0, 0, 0],
[1, 1, 1, 1],
[0, 0, 1, 0],
[0, 0, 1, 0],
], float)
img_name = 'maps/T.jpg'

cv2.imshow(img_name, a)
cv2.waitKey()

a = np.array(a, dtype=np.uint8)*255
cv2.imwrite(img_name, a)

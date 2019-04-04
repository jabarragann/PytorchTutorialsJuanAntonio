import cv2
import numpy as np

print("CV2 test code")


img  = np.random.random((540,540,3))


cv2.imshow("test Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()



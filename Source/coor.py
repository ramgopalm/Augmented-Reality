import cv2
import numpy as np

source = cv2.imread('tennis_2.jpg')
template = cv2.imread('tennis.jpg')

result = cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED)
y,x = np.unravel_index(result.argmax(), result.shape)
center= (int(x),int(y))
color= (200,100,100)
cv2.circle(source,center,2,color,-1)
print x,y
cv2.imshow('Window1',source)
cv2.waitKey(0)
cv2.destroyAllWindows()

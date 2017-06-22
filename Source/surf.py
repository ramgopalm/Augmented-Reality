import cv2
import numpy

image_main =cv2.imread('michael_jordan.jpg')
image_small =cv2.imread('temp.png')

small_one = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
main_one = cv2.cvtColor(image_main, cv2.COLOR_BGR2GRAY)
hessian_threshold = 6000
detector = cv2.SIFT(hessian_threshold)
hkeypoints,hdescriptors = detector.detectAndCompute(main_one,None)
nkeypoints,ndescriptors = detector.detectAndCompute(small_one,None)

hrows = numpy.array(hdescriptors, dtype = numpy.float32)
nrows = numpy.array(ndescriptors, dtype = numpy.float32)
rowsize = len(hrows[0])
samples = hrows
responses = numpy.arange(len(hkeypoints), dtype = numpy.float32)
knn = cv2.KNearest()
knn.train(samples,responses)

for i, descriptor in enumerate(nrows):
    descriptor = numpy.array(descriptor, dtype = numpy.float32).reshape((1, rowsize))
    
    retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 1)
        
    res, dist =  int(results[0][0]), dists[0][0]
    
    color = (200, 100, 100)
    
    x,y = hkeypoints[res].pt
    center = (int(x),int(y))
    cv2.circle(image_main,center,2,color,-1)
    
    
    x,y = nkeypoints[i].pt
    center = (int(x),int(y))
    cv2.circle(image_small,center,2,color,-1)

cv2.imshow('Window',image_main)
cv2.imshow('Window1',image_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

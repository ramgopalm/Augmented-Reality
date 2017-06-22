import cv2
from cv2 import cv
import numpy
import PIL
from PIL import Image
from PIL import ImageChops # used for multiplying images

def surf_func():
    image_main =cv2.imread('tennis_1.jpg')
    image_small =cv2.imread('tennis.jpg')

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
    sum_x = 0
    sum_y = 0
    count = 0
    for i, descriptor in enumerate(nrows):
        descriptor = numpy.array(descriptor, dtype = numpy.float32).reshape((1, rowsize))
        
        retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 1)
            
        res, dist =  int(results[0][0]), dists[0][0]
        
        color = (250, 11, 11)
        
        x,y = hkeypoints[res].pt
        center = (int(x),int(y))
        sum_x = sum_x + x
        sum_y = sum_y + y
        cv2.circle(image_main,center,2,color,-1)
        count = count + 1
        
        x,y = nkeypoints[i].pt
        center = (int(x),int(y))
        cv2.circle(image_small,center,2,color,-1)
    
    
    cv2.circle(image_main,(int(sum_x) / count, int(sum_y) / count), 20,(200, 80, 80),-1)
    cv2.imshow('Window',image_main)
    cv2.imshow('Window1',image_small)
    cv2.waitKey(0)
    #return (int(sum_x) / count, int(sum_y) / count)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""def rect():
    method = cv.CV_TM_SQDIFF_NORMED

    # Read the images from the file
    small_image = cv2.imread('temp.png')
    large_image = cv2.imread('michael_jordan.jpg')

    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows,tcols = small_image.shape[:2]
    return (int(MPx+MPx+tcols) / 2 , int(MPy+MPy+trows) / 2)
    # Step 3: Draw the rectangle on large_image
    #cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

    # Display the original image with the rectangle around the match.
    #cv2.imshow('output',large_image)

    # The image is only displayed if we call this
    #cv2.waitKey(0)"""
    
surf_func()
    # open images


"""def black_onto(img1, img2, k):  
    # create blank white canvas to put img2 onto
    resized = Image.new("RGB", img1.size, "white")

        # define where to paste mask onto canvas
    img1_w, img1_h = img1.size
    img2_w, img2_h = img2.size    
        #box = (img1_w/2-img2_w/2, img1_h/2-img2_h/2, img1_w/2-img2_w/2+img2_w, img1_h/2-img2_h/2+img2_h)
    #box = (23,11)
        # multiply new mask onto image
    resized.paste(img2, k)
    return ImageChops.multiply(img1, resized)




    # this gives the output image shown above
cap = cv2.VideoCapture(0)
while(True):
        # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('tes.jpg', frame)
    k = rect()
    print k
    #im=Image.open('tes.jpg')
    painting = Image.open("tes.jpg")
    mask     = Image.open("home.jpg")

    out = black_onto(painting, mask,k)

        
        #im.show()
    out.save('hel.jpg')
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('tes3.jpg', im)
    ty=cv2.imread('hel.jpg')
        #p=getPixel(ty,100,150)
        # Display the resulting frame
    cv2.imshow('frame',ty)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
cap.release()"""
cv2.destroyAllWindows()





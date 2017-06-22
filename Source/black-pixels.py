import PIL
from PIL import Image
from PIL import ImageChops # used for multiplying images
import cv2

# open images


def black_onto(img1, img2):  
    # create blank white canvas to put img2 onto
    resized = Image.new("RGB", img1.size, "white")

    # define where to paste mask onto canvas
    img1_w, img1_h = img1.size
    img2_w, img2_h = img2.size    
    #box = (img1_w/2-img2_w/2, img1_h/2-img2_h/2, img1_w/2-img2_w/2+img2_w, img1_h/2-img2_h/2+img2_h)
    box = (23,11)
    # multiply new mask onto image
    resized.paste(img2, box)
    return ImageChops.multiply(img1, resized)



# this gives the output image shown above
cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('tes.jpg', frame)
    im=Image.open('tes.jpg')
    painting = Image.open("tes.jpg")
    mask     = Image.open("home.jpg")

    out = black_onto(painting, mask)

    
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
cap.release()
cv2.destroyAllWindows()

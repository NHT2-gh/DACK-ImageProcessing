import cv2
import math

img = cv2.imread('images/pc.jpg')
scale = float(1500)/(img.shape[0]+img.shape[1])
print(scale)
# img = cv2.resize(img,(int(img.shape[1]*scale), int(img.shape[0] * scale)))

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
adjust_v = (img_hsv[:,:,2].astype("uint") + 5)*3
adjust_v = ((adjust_v > 255)*255 + (adjust_v <= 255) * adjust_v).astype("uint8")
img_hsv[:,:,2] = adjust_v
img_soft = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
img_soft = cv2.GaussianBlur(img_soft, (51,51),0)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2. equalizeHist(img_gray)
invert = cv2.bitwise_not(img_gray)
blur = cv2.GaussianBlur(invert, (21,21),0)
invertBlur = cv2.bitwise_not(blur)
sketch = cv2.divide(img_gray, invertBlur, scale = 265.0)
sketch = cv2.merge([sketch, sketch, sketch])

img_water = ((sketch/255.0) * img_soft).astype("uint8")
cv2.imshow("final",img_water)
cv2.waitKey(0)
import cv2
import matplotlib.pyplot as plt 
import numpy as np

image_name = "circles.png"

trainer = cv2.imread("C:\\users\\USER\\Documents\\" + image_name)
trainer = cv2.cvtColor(trainer, cv2.COLOR_BGR2RGB)

img = cv2.imread("C:\\users\\USER\\Documents\\" + image_name)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
main = cv2.cvtColor(hsv,cv2.COLOR_BGR2GRAY)

lower_value = np.array([25,150,50])
upper_value = np.array([35,255,255])  

mask = cv2.inRange(hsv,lower_value,upper_value)
contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)

for q in contours:
    x,y,w,h = cv2.boundingRect(q)
    cv2.rectangle(trainer,(x,y),(x+w,y+h),(255,0,255),2)

fig = plt.figure()
fig.set_figwidth(15)
fig.set_figheight(10)

images = [trainer,mask]
for i in range(len(images)):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')

cv2.waitKey(0)

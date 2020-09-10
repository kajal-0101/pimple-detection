# import the necessary packages
import argparse
import numpy as np
import imutils
import cv2

# pimple counter
p = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# load the image 
im = cv2.imread(args["image"])

# crop region of interest
r = cv2.selectROI("Select Region of Interest", im)
image = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]


# splitting the image
chans = cv2.split(image)
gray = chans[1]

adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(adaptive.copy(), kernel,2)


# find contours in the threshold image
cnts = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	if(cv2.contourArea(c) > 20 and cv2.contourArea(c) < 150):
		x,y,w,h = cv2.boundingRect(c)
		roi = image[y:y+h,x:x+w]
		hsr = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		color = cv2.mean(hsr)
		if (int(color[0]) < 10 and int(color[1]) > 70 and int(color[2]) > 90):
			(a,b),radius = cv2.minEnclosingCircle(c)
			if (int(radius) < 20):
				x,y,w,h = cv2.boundingRect(c)
				cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),0)
				p += 1

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imwrite("image.jpg", image)

print("The are {} pimple in the image" .format(p))


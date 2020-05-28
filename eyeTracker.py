# USAGE
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml --video video/adrian_eyes.mov
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml

# import the necessary packages
from et import ET
import imutils
import cv2

#replace parameters with file location of recognition libraries if different
et = ET("cascades/haarcascade_frontalface_default.xml", "cascades/haarcascade_eye.xml") #recognition libraries
camera = cv2.VideoCapture(0) #initializes camera


#function that identifies eye and resizes frame to only one eye
def resizeFrame():
	# grab the current frame
	(grabbed, frame) = camera.read()

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# use tracking function to get the dimensions and locations of where to put the rectangles for eyes and face
	rects_f, rects_e = et.track(gray)


	# draw the rectangles in the full sized camera view
	for r in rects_f:
		cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
	for r in rects_e:
		cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

	#makes a copy of the frame to manipulate and resize
	frameClone = frame.copy()

	#if it identifies and tracks eyes, collapse the box to only include the first eye
	# if there is an eye detected
	if len(rects_e) > 0:
		eye1 = rects_e[0]
		eye1 = rolling_average(eye1) #rolling average makes the box smoother but runs into issues when "eye1" switches between right and left
		# if the height and width of the first eye is greater than 0 (aka exists)
		if eye1[3] > 0 and eye1[2] > 0:
			#resize frame to just the dimensions of eye tracking box
			#adding and subtracting percentage of frame reduce the frame to include as little area around the eye as possible
			ypercent = 0#int((eye1[3] - eye1[1]) * .25)
			xpercent = 0#int((eye1[2] - eye1[0]) * .13)
			frameClone = frameClone[eye1[1]+ypercent:eye1[3]-ypercent, eye1[0]+xpercent:eye1[2]-xpercent]
			#open a window showing only the first eye
			#cv2.imshow("Only eye", frameClone)
	#otherwise show the whole face
	else:
		frameClone = []

	#continuously show entire frame in separate window
	cv2.imshow("Whole camera", frame)
	#returns the frame cropped to only one eye
	return frameClone



recents = []
#takes in a list of integers (x, y, width, and height) and computes a rolling average for the previous 5 frames
def rolling_average(current):
	#add the current frame onto the end of the list
	recents.append(current)
	if len(recents) > 5:
		#limit the list length to the previous 5 frames
		recents.pop(0)

	avg_vals = [0, 0, 0, 0]
	#adds all the x values, y values, etc in one list (avg_vals)
	for l in recents:
		for i in range(0, 4):
			avg_vals[i] += l[i]

	#computes the average and makes it an acceptable type (a rounded integer)
	for i in range(0, len(avg_vals)):
		avg_vals[i] = int(round(avg_vals[i]/len(recents)))

	return avg_vals



# keeps looping while the program is running
while True:

	#function that resizes frame from original to just one eye
	eyeFrame = resizeFrame()

	#maybe use value as a percentage of the highest so it adjusts for high and low light conditions
	#if an eye is detected, try to find the pupil
	if len(eyeFrame) > 0:
		#converts eye frame to grayscale and applies Gaussian blur to reduce noise
		gray_eye = cv2.cvtColor(eyeFrame, cv2.COLOR_BGR2GRAY)
		gray_eye = cv2.GaussianBlur(gray_eye, (9, 9), 0) #(7,7)


		#only identifies darkest parts of frame (trying to find pupil)
		threshold = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
		#finds the contours of the threshold
		contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#sorts the contours by area with largest area first
		contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)

		for cnt in contours:
			#draws the contours on the original color eye
			cv2.drawContours(eyeFrame, [cnt], -1, (0, 0, 255), 1)
			#stops the loop after the first one so only the contour with the biggest area is drawn
			break

		cv2.imshow("Only eye", eyeFrame)
		cv2.imshow("grayscale eye", gray_eye)
		cv2.imshow("threshold", threshold)



	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
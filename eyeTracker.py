# USAGE
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml --video video/adrian_eyes.mov
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml

# import the necessary packages
from et import ET
import imutils
import argparse
import cv2

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--eye", required = True,
	help = "path to where the eye cascade resides")
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the eye tracker
et = ET(args["face"], args["eye"])

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])
'''
#since arguments are always the same I took out the important parts
#from commented section above
et = ET("../cascades/haarcascade_frontalface_default.xml", "eye ../cascades/haarcascade_eye.xml")
camera = cv2.VideoCapture(0)

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	#if args.get("video") and not grabbed:
		#break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# use tracking function to get the dimensions and locations of where to put the rectangles for eyes and face
	f_rects, e_rects = et.track(gray)
	print(e_rects) #empty always --> fix
	print(f_rects) #empty always --> fix

	# draw the rectangles
	for r in f_rects:
		cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
	for r in e_rects:
		cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

	#makes a copy of the frame to manipulate and resize
	frameClone = frame.copy()


	#if it identifies and tracks eyes, collapse the box to only include the first eye
	# if the height and width of the first eye is greater than 0 (aka exists)
	try:
		eye1 = e_rects[0]
		if eye1[3] > 0 and eye1[2] > 0:
			#resize frame to just the dimensions of eye tracking box (put rolling average in here)
			frameClone = frameClone[eye1[1]:eye1[1] + (eye1[3]-f_rects[0][1]),
			eye1[0]:eye1[0] + (eye1[4]-f_rects[0][0])]
			cv2.imshow("Only eye", frameClone)
			print("just eye")
	#otherwise show the whole face
	except:
		frameClone = frame.copy()
		print("normal")


	cv2.imshow("Whole camera", frame)
	# show the frame

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
import cv2
import sys
import urllib.request
from em_model import EMR
import numpy as np

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# initialize the cascade
cascPath = "haarcascade_files/haarcascade_frontalface_default.xml"

# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')  

# Initialize object of EMR class
network = EMR()
network.build_network()

font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []

# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3 ,minNeighbors = 5)

    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(image, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

# initialize the list of image URLs to download
urls = [
	"https://image.shutterstock.com/image-photo/portrait-young-beautiful-cute-cheerful-260nw-666258808.jpg",
	"https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/beautycheck/english/kindchenschema/kindfrau_c/kindfrau_c_100_gr.jpg",
	"https://image.shutterstock.com/image-photo/portrait-african-woman-serious-expression-260nw-140713156.jpg",
]

# loop over the image URLs
for url in urls:
	# download the image URL and display it
	print ("downloading %s" % (url))
	image = url_to_image(url)
	
	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)

	# Read the image
	#image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
	
	print("Found {0} faces!".format(len(faces)))		
		
	

	# Again find haar cascade to draw bounding box around face
    

    # compute softmax probabilities
	result = network.predict(format_image(image))
	if result is not None:
		# write the different emotions and have a bar to indicate probabilities for each class
		for index, emotion in enumerate(EMOTIONS):
			cv2.putText(image, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
			cv2.rectangle(image, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

		# find the emotion with maximum probability and display it
		maxindex = np.argmax(result[0])
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,EMOTIONS[maxindex],(10,360), font, 2,(255,255,255),2,cv2.LINE_AA) 
		face_image = feelings_faces[maxindex]
		print(face_image[:,:,3])
		
		
	if not len(faces) > 0:
		# do nothing if no face is detected
		a = 1
	else:
        # draw box around face with maximum area
		max_area_face = faces[0]
		for face in faces:
			if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
				max_area_face = face
		face = max_area_face
		(x,y,w,h) = max_area_face
		image = cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(255,0,0),2)
		
		cv2.imshow("Faces found", cv2.resize(image,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
		cv2.waitKey(0)
		
		
	


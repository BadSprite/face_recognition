import face_recognition
import picamera
import numpy as np
known_face_encodings = []
names = []
def load_face_encoding(name, file_name):
   image = face_recognition.load_image_file(file_name)
   face_encoding = face_recognition.face_encodings(image)[0]
   known_face_encodings.append(face_encoding)
   names.append(name)
# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.
camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)
# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")
load_face_encoding("Wang Lei", "wanglei.jpg")
load_face_encoding("Obama", "obama.jpg")
load_face_encoding("Messi","messi.jpg")
load_face_encoding("Ronaldo","ronaldo.jpg")
load_face_encoding("Biden", "biden.jpg")
# Initialize some variables
face_locations = []
face_encodings = []
while True:
   print("Capturing image.")
   # Grab a single frame of video from the RPi camera as a numpy array
   camera.capture(output, format="rgb")
   # Find all the faces and face encodings in the current frame of video
   face_locations = face_recognition.face_locations(output)
   print("Found {} faces in image.".format(len(face_locations)))
   face_encodings = face_recognition.face_encodings(output, face_locations)
   # Loop over each face found in the frame to see if it's someone we know.
   for face_encoding in face_encodings:
       # See if the face is a match for the known face(s)
       matches = face_recognition.face_distance(known_face_encodings, face_encoding)
       name = "<Unknown Person>"
       min_distance = min(matches)
       if min_distance < 0.6:
           i = matches.argmin()
           name = names[i]
       print("I see someone named {}!".format(name))

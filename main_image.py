import cv2
import numpy as np
import argparse
import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#Emotions
emo = ['Angry', 'Surprised','Sad', 'Happy']
string = 'No sentiment'
############## Spanish version #################
#emo = ['Bravo', 'Sorprendido','Triste', 'Feliz']
#string = 'Sin emocion'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Process some image to find sentiment in faces (if any)')
ap.add_argument("-f", "--file_name", required=False, default="imgs/angry.jpg", help="path to image")
args = vars(ap.parse_args())

file_name = args["file_name"]

# Instantiates a client
#vision_client = vision.Client() # old way
vision_client = vision.ImageAnnotatorClient()

#image = vision_client.image(filename=file_name) #old
# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    file_name)

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#faces = image.detect_faces(limit=20) #old call
faces = vision_client.face_detection(image=image).face_annotations
print ('Number of faces: ', len(faces))

img = cv2.imread(file_name)

for face1 in faces:
    # with open('data.txt', 'w') as outfile:
    #     json.dump(face1, outfile)
    # #print(json.dump(face1))
   
   #old calls
   # x = face1.fd_bounds.vertices[0].x_coordinate
   # y = face1.fd_bounds.vertices[0].y_coordinate
   # x2 = face1.fd_bounds.vertices[2].x_coordinate
   # y2 = face1.fd_bounds.vertices[2].y_coordinate

   x = face1.bounding_poly.vertices[0].x 
   y = face1.bounding_poly.vertices[0].y 
   x2 = face1.bounding_poly.vertices[2].x 
   y2 = face1.bounding_poly.vertices[2].y 


   cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

   #sentiment = [face1.anger.value,face1.surprise.value,face1.sorrow.value,face1.joy.value]
   sentiment = [face1.anger_likelihood,face1.surprise_likelihood,face1.sorrow_likelihood,face1.joy_likelihood]
   for item, item2 in zip(emo, sentiment):
    print (item, ": ", item2)

#    if not (all( item == 'VERY_UNLIKELY' for item in sentiment) ):

#         if any( item == 'VERY_LIKELY' for item in sentiment):
#             state = sentiment.index('VERY_LIKELY')
#             # the order of enum type Likelihood is:
#             #'LIKELY', 'POSSIBLE', 'UNKNOWN', 'UNLIKELY', 'VERY_LIKELY', 'VERY_UNLIKELY'
#             # it makes sense to do argmin if VERY_LIKELY is not present, one would espect that VERY_LIKELY
#             # would be the first in the order, but that's not the case, so this special case must be added
#         else:
#             state = np.argmin(sentiment)
    
        # string = emo[state]

   state = np.argmax(sentiment)
   string = emo[state]
   cv2.putText(img,string, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

cv2.imshow("Video", img)
cv2.waitKey(0)
cv2.imwrite('output/output_'+string+'.jpg',img)

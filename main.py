import cv2
import numpy as np
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
#Emotions
emo = ['Angry', 'Surprised','Sad', 'Happy']

############## Spanish version #################
#emo = ['Bravo', 'Sorprendido','Triste', 'Feliz']
#string = 'Sin emocion'

#from google.oauth2 import service_account
#credentials = service_account.Credentials.from_service_account_file('VisionApp-9cb3e521631b.json')

# Instantiates a client  
#vision_client = vision.Client()  #old call

vision_client = vision.ImageAnnotatorClient()
cv2.imshow('Video', np.empty((5,5),dtype=float))
compressRate = 1
while cv2.getWindowProperty('Video', 0) >= 0:
    video_capture = cv2.VideoCapture(0)
    ret, img = video_capture.read()
    img = cv2.resize(img, (0,0), fx=compressRate , fy=compressRate )

    ok, buf = cv2.imencode(".jpeg",img)
    #image = vision_client.image(content=buf.tostring())  #OLD call
    image = types.Image(content = buf.tostring())


    #faces = image.detect_faces(limit=20) #OLD call
    faces = vision_client.face_detection(image=image).face_annotations
    print ('Number of faces: ', len(faces))
    for i in range(0,len(faces)):
        face1 = faces[i]
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

        string = 'No sentiment'

        # if not (all( item == 'VERY_UNLIKELY' for item in sentiment) ):
        #     if any( item == 'VERY_LIKELY' for item in sentiment):
        #         state = sentiment.index('VERY_LIKELY')
        #         # the order of enum type Likelihood is:
        #         #'LIKELY', 'POSSIBLE', 'UNKNOWN', 'UNLIKELY', 'VERY_LIKELY', 'VERY_UNLIKELY'
        #         # it makes sense to do argmin if VERY_LIKELY is not present, one would espect that VERY_LIKELY
        #         # would be the first in the order, but that's not the case, so this special case must be added
        #     else:
        #         state = np.argmin(sentiment)

        #     string = emo[state]

        state = np.argmax(sentiment)
        string = emo[state]

        cv2.putText(img,string, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Video", img)
    cv2.waitKey(1)
    video_capture.release()

# When everything is done, release the capture
cv2.destroyAllWindows()

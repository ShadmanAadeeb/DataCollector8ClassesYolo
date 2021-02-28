import numpy as np
import cv2
import pyautogui as pg 
import keyboard
import os

currentDirectoryNo=0
noOfImagesInCurrentDirectory=-1
kernel = np.ones((5,5),np.uint8)

#Getting the camera
cap = cv2.VideoCapture(1)

#making the yolo nn architecture
net = cv2.dnn.readNet("./yolov4-tiny-hand_best.weights","./yolov4-tiny-testing.cfg")
#Getting reference to the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#Making an array to contain the name of objects in final layer
classes = ["hand"]



while(True):

    #**********************Dealing with the key presses starts************************#
    if keyboard.is_pressed('s'):
        backGroundMaskMaking= not backGroundMaskMaking 
        cv2.destroyAllWindows()
    elif keyboard.is_pressed('1'):
        currentDirectoryNo=1
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=  len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('2'):
        currentDirectoryNo=2
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('3'):
        currentDirectoryNo=3
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('4'):
        currentDirectoryNo=4
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('5'):
        currentDirectoryNo=5
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('6'):
        currentDirectoryNo=6
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('7'):
        currentDirectoryNo=7
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('8'):
        currentDirectoryNo=8
        i=currentDirectoryNo        
        noOfImagesInCurrentDirectory=len(os.listdir("./Gesture"+str(i))) 
    elif keyboard.is_pressed('0'):
        currentDirectoryNo=0
        i=currentDirectoryNo
        noOfImagesInCurrentDirectory=-1
        

    #**********************Dealing with the key presses ends************************#


    #Getting image from the camera
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    height, width, channels = img.shape
    


    #Processing the image (i.e making the blob)
    #dividing by 255, reshaping into 288 by 288
    blob = cv2.dnn.blobFromImage(img, 0.00392 , (288, 288), (0, 0, 0), True, crop=False)
    
    #Passing the blob to the neural network
    net.setInput(blob)
    #Collecting the feature maps from the output layers
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            #The three lines below is used for finding confidence value
            scores=detection[5:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence>0.3:
                #It means we are considering a detected object                    
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                                        
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # The boxes array contain the detected boxes
        
        # We further get the indexes of boxes after Non Max Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        
        font = cv2.FONT_HERSHEY_PLAIN
        
        
        #The boxes and indexes have been obtained,so we perform background subtraction
        
        #img=img-backGroundMask
        
        
        #Now drawing the boxes from the images
        
        for i in range(len(boxes)):
            if i in indexes:
                try:
                    x, y, w, h = boxes[i]
                    #print("x1,y1 is= "+str(x)+", "+str(y))
                    label = str(classes[class_ids[i]])
                    #print("Label ",label)
                    color = (244,0,0)
                    
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, "Gesture:"+str(currentDirectoryNo), (10, 50), font, 3, color, 1)
                    cv2.putText(img, ",NoOfImages:"+str(noOfImagesInCurrentDirectory), (250, 50), font, 3, color, 1)

                    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    

                    croppedImg = img[y:y+h, x:x+w]         
                    #croppedImg = cv2.Canny(croppedImg,100,200)
                    #croppedImg = cv2.dilate(croppedImg,kernel,iterations = 1)
                    croppedImg = cv2.resize(croppedImg, (100,100), interpolation = cv2.INTER_AREA)
                    cv2.imshow("Cropped Image", croppedImg)
                    if(currentDirectoryNo!=0):
                        cv2.imwrite("./Gesture"+str(currentDirectoryNo)+"/"+str(noOfImagesInCurrentDirectory)+".jpg", croppedImg) 
                        noOfImagesInCurrentDirectory=noOfImagesInCurrentDirectory+1
                	#cv2.imshow("Cropped", croppedImg)

                except:
                	pass
        
        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel); 
        cv2.imshow("Image", img)
    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



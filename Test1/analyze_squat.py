#TODO launch on GIThub, make opencv run faster, work on deadlift bot and other AIs

#Import Dependencies
import mediapipe as mp
import cv2
import numpy as np

import csv #Store coordinates inside of a local file on the computer
import os
import pandas as pd #If it gives a warning that it can't find this module, run through terminal using curlenv
import pickle

from datetime import datetime

import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = 'mongodb+srv://juancavallin:password@cavallin.1qowuux.mongodb.net/?retryWrites=true&w=majority'

# Create a new client and connect to the server
cluster = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
cloud_database = cluster["UserData"]
user_landmarks = cloud_database["User_Landmarks"]
user_videos = cloud_database["User_Videos"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Join upload folder and processed_videos folder paths
#now = os.getcwd()
#path = os.path.join(now, "C:/Users/juanc/OneDrive/Desktop/PoseNet Tutorial/Project")
#It is normally bad practice to make variables global, but I will leave them like this for now in case I need another function to access them
global reps; global stage; global ascending; global counter; global error_message; 

#Import ML models

with open('Squat_Test_Models/Squat_Landmark_RC.pkl', 'rb') as f: 
    rc_model = pickle.load(f)
with open('Squat_Test_Models/Squat_Landmark_RF.pkl', 'rb') as f: 
    rf_model = pickle.load(f)
with open('Squat_Test_Models/Squat_Landmark_GB.pkl', 'rb') as f: 
    gb_model = pickle.load(f)

form = { "Depth" : -1000.0 } #Shortened
#files = os.listdir('Good_Form_Videos') #Will store all of the uploaded video files currently 
files = ["C:/Users/juanc/OneDrive/Desktop/PoseNet Tutorial/Project/Good_Form_Videos/high-bar.gif"]
#files = ['Good_Form_Videos/high-bar.gif'] #Will store

def calculate_angle(a, b, c):
    a = np.array(a) #First joint (shoulder)
    b = np.array(b) #Middle joint (elbow)
    c = np.array(c) #last joint/ endpoint (wrist)
    
    #Calculate angle. b[0] is the x value and b[1] is the y value for that joint
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0: #Keep angle at max 180 degrees, MAY NEED TO CHANGE FOR SQUAT
        angle = 360 - angle
    
    return angle
  
def process_video(name, squat_video, date, recorder):    
    reps = 0; counter = 0; stage = 'none'; ascending = False; error_message = ''  #Reset variables
    #With keyword is used as a neater way of writing try catch. experiment with detection values to capture more/ less sensitively. Feed is stored under pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 

        while squat_video.isOpened:
            ret, frame = squat_video.read() 
            
            if ret:
                squat_video.set(cv2.CAP_PROP_POS_FRAMES, counter)
                counter += 1 #Powerlifter 45-635.mp4 runs at 30fps. I have decided that the most accurate way of capturing data is to capture every frame instead of every 5-6 frames
                
                #Convert image from BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #image = image_resize(image, 800, 800)
                image.flags.writeable = False #Save memory
                
                #Make detection
                results = pose.process(image) #stores the position of each landmark - joint, eyes, nose, etc.
                #Turn image back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)     
                #Extract Landmarks 
                try:
                
                    landmarks = results.pose_landmarks.landmark
                    
                    world_landmarks = results.pose_world_landmarks.landmark #For future use in designing ML algorithms      
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    #calc angles
                    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

                    #I changed this to only continue if there is an error detecting the landmarks

                
                    #Used to count # of reps
                    depth = (l_knee[1] - l_hip[1] + r_knee[1] - r_hip[1]) * 50 #pos = distance below parallel
                    #TODO calculate real life distance instead of pixel distance based on height, ask Namit about the idea    
                    #Squat Counter logic Track info on the way up and down
                    rep_frames = []    
                    if (depth - form['Depth']) > 2.0:
                        rep_frames.append(counter)    
                        #if left_knee_angle > 100 and right_knee_angle > 100:                  
                        if stage == 'down':
                            ascending = True
                            if left_knee_angle < 100 and right_knee_angle < 100:
                                error_message += 'depth should be at least parallel'
                                #highlight_error() #Depth not low enough TODO adjust error message. Maybe change it so that the errors are viewed afterwards                    
                        stage = 'up'               
                        
                    elif (depth - form['Depth']) < -2.0:
                            #if left_knee_angle < 75 and right_knee_angle < 75:
                            rep_frames.append(counter)
                            stage = 'down'
                            ascending = False
                    elif depth > 8:
                            stage = 'rest'                    
                    form['Depth'] = depth
                    
                    #Changed so that it detects the rep completed at the top of the movement. This allows me to identify how many frames the rep lasted
                    if ascending == True and stage != 'up':
                        #form_reps.insert(reps, rep_frames)
                        reps += 1
                        print('Rep:', reps)
                        #Print form and reset
                        print(form)
                        rep_frames = []
                        ascending = False                   
                
                    #Create two databases for testing: one with all of the landmarks tested and one with only the pose 
                    #TODO find videos for data and test world vs normalized landmark coordinates to see if there is a difference (might not be, but I think it has the center at the hips)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)) 
                    
                    row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
                    X = pd.DataFrame([row])
                    rf_class = rf_model.predict(X)[0]
                    rf_prob = rf_model.predict_proba(X)[0]
                    
                    rc_class = rc_model.predict(X)[0]
                    rc_prob = rc_model.decision_function(X)[0] 
                    
                    gb_class = gb_model.predict(X)[0]
                    gb_prob = gb_model.predict_proba(X)[0]   
                    #Print pose values into other algorithm
                    #row2 = row[50:]
                    
                    
                    
                                    
                    #render_info(image, 'Feed') I will only render the probabilities for now 
                    cv2.putText(image, 'RF', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(rf_class), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'PROB1', (50, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(rf_prob[np.argmax(rf_prob)], 2)), (65, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'RC', (150, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(rc_class), (165, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    
                    cv2.putText(image, 'PROB2', (185, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(rc_prob, 2)), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'GB', (250, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(gb_class), (265, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'PROB3', (285, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(gb_prob[np.argmax(gb_prob)], 2)), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except: #This will occur when there are no landmarks detected
                    pass                
                cv2.imshow('Mediapipe Feed', image) #display the new image with landmark and connections marked instead of frame
                
                #Save frame into video with drawings on top 
                recorder.write(image)
                
                landmark_post = {"_id": date + name + '_' + str(counter), "rf_prob" : round(rf_prob[np.argmax(rf_prob)], 2), 
                                 "rc_prob" : round(rc_prob, 2), "gb_prob" : round(gb_prob[np.argmax(gb_prob)], 2),
                                 "stage" : stage, "reps" : reps, "landmarks:" : row }
                
                user_landmarks.insert_one(landmark_post)
                
            else:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                break
            
        squat_video.release() 
        recorder.release() 
        cv2.destroyAllWindows()
        print('Video has been processed')

#Create window beforehand as a safe measure
cv2.namedWindow('Mediapipe Feed')
#For now this code saves each video as a separate video file. Maybe I should instead save all the videos in a single file. May also want to save as mp4 instead of avi
for file in files: 
    squat_video = cv2.VideoCapture(file)
    #Width and height based on video used 
    width = int(squat_video.get(cv2.CAP_PROP_FRAME_WIDTH)); 
    height = int(squat_video.get(cv2.CAP_PROP_FRAME_HEIGHT)); 
    fps = int(squat_video.get(cv2.CAP_PROP_FPS)); 
    file_name = os.path.splitext(os.path.basename(file))[0]; date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = 'Processed_Videos/' + date + file_name + '.mp4'
    recorder = cv2.VideoWriter(str(name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    process_video(file_name, squat_video, date, recorder)
#TODO come up with way of tracking ID for MongoDB. Submit video file through here to Mongo as well
#TODO Make server display processed video after hitting submit. Orgranize files between Test1 and Project



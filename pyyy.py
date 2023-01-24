import cv2
import mediapipe as mp
import pyautogui
from deepface import DeepFace
from threading import Thread
from customtkinter import *
from PIL import Image,ImageTk

from keras.preprocessing import image
import numpy as np
global frame
global predicted_emotion
cam = cv2.VideoCapture(0)
_, frame = cam.read()
app = CTk()
predicted_emotion = "None"
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
CamFrame = CTkFrame(app)
label =CTkLabel(CamFrame)
infoFrame = CTkFrame(app)
#textLabel = CTkLabel(infoFrame,wraplength=400, justify="center")
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

numLabels = []
for idx,emotion in enumerate(emotions):
    textLabel= CTkLabel(infoFrame,wraplength=100, justify="center",text=emotion)
    numLabel = CTkLabel(infoFrame,wraplength=100, justify="center",text=0)
    numLabels.append(numLabel)
    textLabel.grid(row=0,column=idx)
    numLabel.grid(row=1,column=idx)
predictionLabel = CTkLabel(infoFrame,wraplength=300, justify="center",text="Predicted dominant Emotion",text_color="Red") # ,text_font=("Arial", 15)
predictionLabel.grid(row=0,column=idx+1)

prediction = CTkLabel(infoFrame,wraplength=100, justify="center",text="None",text_color="green") # ,text_font=("Arial", 15)
prediction.grid(row=1,column=idx+1)
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

def show_frames():
    global frame, predicted_emotion
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    #frame = rescale_frame(frame,percent=150)
    #frame = cv2.scaleAdd()
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h,frame_w,_ = frame.shape
    faces_detected = face_haar_cascade.detectMultiScale(rgb_frame)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = rgb_frame[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        # img_pixels = np.asarray(roi_gray)
        # img_pixels = np.expand_dims(img_pixels, axis=0)
        # img_pixels /= 255

        cv2.putText(rgb_frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # if landmark_points:
    #     landmarks = landmark_points[0].landmark
    #     for id,landmark in enumerate(landmarks[468:478]):
    #         x = int(landmark.x * frame_w)
    #         y = int(landmark.y * frame_h)
    #         cv2.circle(rgb_frame,(x,y),3,(0,255,0))
    
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20,show_frames)
def analyze():
    global frame
    global predicted_emotion
    while True:
        
        try:
            result = DeepFace.analyze(frame,actions=['emotion'])
            
            #print(result['emotion'])
            for idx,value in enumerate(result['emotion'].values()):
                numLabels[idx].configure(text=round(value,5))
            predicted_emotion = result["dominant_emotion"]
            prediction.configure(text=result["dominant_emotion"])
            #textLabel.configure(text=[f"{key} : {value}" for key,value in result['emotion'].items()])
            #print(result)

        except Exception as e:
            pass
            #print(e)
            #print(" no face det")
            #textLabel.configure(text="no face")


if __name__ == "__main__":
    t1 = Thread(target = show_frames)
    t1.setDaemon(True)
    
    t2 = Thread(target = analyze)
    t2.setDaemon(True)
    t1.start()
    t2.start()

    #show_frames()
    CamFrame.pack(side='top',fill='both')
    infoFrame.pack(fill='both')
    label.pack()
    #textLabel.pack()

    app.geometry("1300x800")
    app.resizable(False, True)
    app.mainloop()
    
    # while True:
        
    #     try:
    #         result = DeepFace.analyze(frame,actions=['emotion'])
    #         # for ind,res in enumerate(result):
    #         #     cv2.putText(frame,f"{res}",(200,200+ind*10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
    #     except:
    #         print("no face")
        


# boiler code


# def runTaskA():
#     label =CTkLabel(app)
#     label.pack()
#     cam = cv2.VideoCapture(0)
#     face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
#     screen_w,screen_h= pyautogui.size()
    
#     global frame
#     def rescale_frame(frame, percent=75):
#         width = int(frame.shape[1] * percent/ 100)
#         height = int(frame.shape[0] * percent/ 100)
#         dim = (width, height)
#         return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
#     while True:
#         _, frame = cam.read()
#         frame = cv2.flip(frame,1)
#         frame = rescale_frame(frame,percent=150)
#         #frame = cv2.scaleAdd()
#         rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         output=face_mesh.process(rgb_frame)
#         landmark_points = output.multi_face_landmarks
#         frame_h,frame_w,_ = frame.shape
#         if landmark_points:
#             landmarks = landmark_points[0].landmark
#             for id,landmark in enumerate(landmarks[468:478]):
#                 x = int(landmark.x * frame_w)
#                 y = int(landmark.y * frame_h)
#                 cv2.circle(rgb_frame,(x,y),3,(0,255,0))
        
#         img = Image.fromarray(rgb_frame)
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)
#         #label.after(20,)
#         #cv2.imshow("Eye Controll",frame)
#         #cv2.waitKey(1)
#         # if cv2.waitKey(1) & 0xFF==ord("q"):
#         #     break
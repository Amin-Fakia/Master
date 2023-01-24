from customtkinter import *
import numpy as np
import cv2
import mediapipe as mp
from threading import Thread
from PIL import Image, ImageTk,ImageDraw
import time
import scipy
import tkinter as tk


set_appearance_mode("Dark")



def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im
# def zoom(img, zoom_factor=2):
#     return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

class VideoCapture:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.image = None
    
    def get_image_imp(self,size=(500,400)):
        success, self.image = self.cap.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return self.image
    def flip_image(self,flip=0):
        return cv2.flip(self.image,flip)
    
    def zoom(self, zoom_factor=1):
        return cv2.resize(self.image, None, fx=zoom_factor, fy=zoom_factor)
        
    def get_image(self,size=(500,400),flip=0):
        
        success, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if flip == 1:
            image = cv2.flip(image,1)
        
        im = Image.fromarray(image)
        im = add_corners(im,10)
        #imgtk = ImageTk.PhotoImage(image=im.resize((500,400)))
        
        test_img = CTkImage(im,size=size)
        return test_img,image
    

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmark 
    return mesh_coord


class App(CTk):

    def zoomVideo(self, image, Iscale=1):
    
        scale=Iscale

        #get the webcam size
        height, width, channels = image.shape

        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*centerX),int(scale*centerY)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height))

        return resized_cropped
    def process_image(self,image,size=(500,400),flip=0,zoom=None):
        if zoom:
            image = self.zoomVideo(image,zoom)
        if flip == 1:
            image = cv2.flip(image,1)      
        im_arr = Image.fromarray(image)
        im_arr = add_corners(im_arr,10)
        return CTkImage(im_arr,size=size),image
    def test_thread(self):
        while True:
            im = self.camera_obj.get_image_imp()
            # # im = zoom(im,self.test_slider.get())
            
            face_array = np.zeros(im.shape,dtype=np.uint8)
            eye_array = np.zeros(im.shape,dtype=np.uint8)
            #im = zoomVideo(im,self.test_slider.get())
            
            imageTest,im= self.process_image(im,(470,370),flip=int(self.FLIP.get()),zoom=self.test_slider.get())

            results = self.face_mesh.process(im)
            
            if results.multi_face_landmarks:
                
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=face_array,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=face_array,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=eye_array,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                        )
                    
            
            if self.SWITCH.get() == "1":
                #pass
                self.cam_label.configure(image=imageTest) # CTkImage(Image.fromarray(im),size=(500,400))
            else:
                self.cam_label.configure(image=self.imageFromArray(self.black_img_array))
            self.face_label.configure(image=self.imageFromArray(face_array))
            self.eye_label.configure(image=self.imageFromArray(eye_array))
            self.cam_label.after(int(1000/120),self.update_camera_frame)

   

    def update_camera_frame(self):
        
       
       
        im = self.camera_obj.get_image_imp()
        # # im = zoom(im,self.test_slider.get())
        
        face_array = np.zeros(im.shape,dtype=np.uint8)
        eye_array = np.zeros(im.shape,dtype=np.uint8)
        #im = zoomVideo(im,self.test_slider.get())
        
        imageTest,im= self.process_image(im,(470,370),flip=int(self.FLIP.get()),zoom=self.test_slider.get())

        results = self.face_mesh.process(im)
        
        if results.multi_face_landmarks:
            
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=face_array,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                self.mp_drawing.draw_landmarks(
                    image=face_array,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image=eye_array,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                    )
                
        
        if self.SWITCH.get() == "1":
            #pass
            self.cam_label.configure(image=imageTest) # CTkImage(Image.fromarray(im),size=(500,400))
        else:
            self.cam_label.configure(image=self.imageFromArray(self.black_img_array))
        self.face_label.configure(image=self.imageFromArray(face_array))
        self.eye_label.configure(image=self.imageFromArray(eye_array))
        self.cam_label.after(int(1000/120),self.update_camera_frame)

        
        
            
            
    def imageFromArray(self,array,size=(470,370)):
        img= Image.fromarray(array)
        img = add_corners(img,10)
        return CTkImage(img,size=size)
        
    def __init__(self):
        super().__init__()
        self.title("Sensor Dashboard")
        self.geometry(f"{1300}x{780}")
        #self.resizable(False,False)
        self.FLIP = StringVar(value="0")
        self.SWITCH = StringVar(value="1")
        self.face_array = None
        self.eye_array = None
        self.imageTest = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        

        self.camera_obj = VideoCapture()

        self.black_img_array = np.zeros((500,400),dtype=np.uint8)
        self.face_img = self.imageFromArray(self.black_img_array)
        
        
        main_frame = CTkFrame(self)
        
        controls_frame = CTkFrame(self,width=250)
        controls_frame.pack_propagate(0)

        

        control_label = CTkLabel(controls_frame,text="Controls",font=("Arial", 20))

        camera_control = CTkFrame(controls_frame)

        cam_switch = CTkSwitch(camera_control,text="Cam On/Off",variable=self.SWITCH,onvalue="1",offvalue="0")
        flip_switch = CTkSwitch(camera_control,text="flip Camera horizontaly",variable=self.FLIP,onvalue="1",offvalue="0")
        

        self.test_slider =CTkSlider(camera_control, from_=3, to=1.01)
        self.test_slider.set(3)
        check_box = CTkCheckBox(controls_frame,text="Test Checkbox")

        control_label.pack(fill="x",side="top",expand=True)
        camera_control.pack()

        cam_switch.pack(fill="x",expand=True,padx=7,pady=7)
        flip_switch.pack(fill="x",expand=True,padx=7,pady=7)
        self.test_slider.pack(fill="x",expand=True,padx=7,pady=7)
        check_box.pack(expand=True)
        

     
        

        

        face_cap = CTkFrame(main_frame,fg_color="transparent")
        self.face_label = CTkLabel(face_cap,image=self.face_img,width=450,height=350,text=None)
        self.face_label.pack(expand=True,fill="both")

        cam_frame = CTkFrame(main_frame,fg_color="transparent")
        self.cam_label = CTkLabel(cam_frame,width=500,height=350,text=None)
        self.cam_label.pack(expand=True,fill="both")


        eye_cap = CTkFrame(main_frame,fg_color="transparent")
        self.eye_label = CTkLabel(eye_cap,width=450,height=350,image=self.face_img,text=None)
        self.eye_label.pack(expand=True,fill="both")

        info_frame = CTkFrame(main_frame)
        self.info_label = CTkLabel(info_frame,width=450,height=350,text="Other Sensors")

        self.info_label.pack(fill="x",side="top")

        
   


        controls_frame.pack(side="left",fill="y",padx=10,pady=10)
        main_frame.pack(fill="both",expand=True,padx=10,pady=10)
        cam_frame.grid(column=0,row=0,padx=10,pady=10)
        face_cap.grid(column=1,row=0,padx=10,pady=10)
        eye_cap.grid(column=0,row=1,padx=10)
        info_frame.grid(column=1,row=1)
        self.update_camera_frame()
        

        



if __name__ == "__main__":
    app = App()
    

    app.mainloop()
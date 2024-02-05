import cv2
import numpy as np
import mediapipe as mp

class IProcessing:
    def init(self, cameraID):
        IProcessing.mpHolistic = mp.solutions.holistic
        IProcessing.mpDraws = mp.solutions.drawing_utils
        IProcessing.mpDrawingStyles = mp.solutions.drawing_styles
        IProcessing.holistic = IProcessing.mpHolistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85)

        IProcessing.mpFaceMesh = mp.solutions.face_mesh
        faceMesh = IProcessing.mpFaceMesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)

        IProcessing.cameraID = cameraID
        IProcessing.Capture = cv2.VideoCapture(IProcessing.cameraID)
        if IProcessing.Capture is None or not IProcessing.Capture.isOpened():
            return False
        else:
            return True
        
    def read_camera(self):
        IProcessing.ret, IProcessing.frame = IProcessing.Capture.read()
        IProcessing.Image = cv2.cvtColor(IProcessing.frame, cv2.COLOR_BGR2RGB)
        IProcessing.h, IProcessing.w, IProcessing.c = IProcessing.Image.shape
        return IProcessing.ret, IProcessing.frame, IProcessing.Image, IProcessing.h, IProcessing.w, IProcessing.c
    
    def stop_camera(self):
        IProcessing.Capture.release()

    def mediapipe_process(self, handCount):
        IProcessing.results = IProcessing.holistic.process(IProcessing.Image)
        if IProcessing.results.face_landmarks:
            IProcessing.mpDraws.draw_landmarks(
                image=IProcessing.Image,
                landmark_list=IProcessing.results.face_landmarks,
                connections=IProcessing.mpFaceMesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=IProcessing.mpDrawingStyles.get_default_face_mesh_contours_style()
            )
        if handCount == 0:
            IProcessing.handResult = IProcessing.results.right_hand_landmarks
        if handCount == 1:
            IProcessing.handResult = IProcessing.results.left_hand_landmarks

        if IProcessing.handResult:
            x_min, x_max, y_min, y_max = IProcessing.drawBox(self, IProcessing.h, IProcessing.w, IProcessing.handResult)
            IProcessing.cropped_image = IProcessing.frame[y_min:y_max, x_min:x_max]
            IProcessing.cropped_image = cv2.flip(IProcessing.cropped_image, 1)

            cv2.rectangle(IProcessing.Image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            IProcessing.mpDraws.draw_landmarks(IProcessing.Image, IProcessing.handResult, IProcessing.mpHolistic.HAND_CONNECTIONS, IProcessing.mpDrawingStyles.get_default_hand_landmarks_style(), IProcessing.mpDrawingStyles.get_default_hand_connections_style())

            return IProcessing.Image, IProcessing.handResult, x_min, x_max, y_min, y_max
        
        return IProcessing.Image, IProcessing.handResult, 0, 0, 0, 0

    def get_retraced_image(self, x_min, x_max, y_min, y_max):
        canvas = np.zeros((x_max - x_min, y_max - y_min, 3), np.uint8)
        landmarkIndex = 0
        if IProcessing.results.face_landmarks:
            for landmarkIndex in range(len(IProcessing.results.face_landmarks.landmark)):
                IProcessing.results.face_landmarks.landmark[landmarkIndex].x = (((IProcessing.results.face_landmarks.landmark[landmarkIndex].x) / 1 * 640) - x_min) / (x_max - x_min)
                IProcessing.results.face_landmarks.landmark[landmarkIndex].y = (((IProcessing.results.face_landmarks.landmark[landmarkIndex].y) / 1 * 480) - y_min) / (y_max - y_min)

            IProcessing.mpDraws.draw_landmarks(
                image=canvas,
                landmark_list=IProcessing.results.face_landmarks,
                connections=IProcessing.mpFaceMesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=IProcessing.mpDrawingStyles
                    .get_default_face_mesh_contours_style())

        for landmarkIndex in range(21):
            IProcessing.handResult.landmark[landmarkIndex].x = (((IProcessing.handResult.landmark[landmarkIndex].x) / 1 * 640) - x_min) / (x_max - x_min)
            IProcessing.handResult.landmark[landmarkIndex].y = (((IProcessing.handResult.landmark[landmarkIndex].y) / 1 * 480) - y_min) / (y_max - y_min)
        IProcessing.mpDraws.draw_landmarks(canvas, IProcessing.handResult, IProcessing.mpHolistic.HAND_CONNECTIONS, IProcessing.mpDrawingStyles.get_default_hand_landmarks_style(), IProcessing.mpDrawingStyles.get_default_hand_connections_style())
        canvas = cv2.resize(canvas, (100, 100))
        canvas = cv2.flip(canvas, 1)   
        return canvas
    
    def get_cropped_image(self):
        IProcessing.cropped_image = cv2.resize(IProcessing.cropped_image, (100, 100))
        return IProcessing.cropped_image

    def drawBox(self, h, w, hand_landmarks):
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for c in range(21):
            x = int(hand_landmarks.landmark[c].x * w)
            y = int(hand_landmarks.landmark[c].y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        return x_min-15, x_max+15, y_min-15, y_max+15    
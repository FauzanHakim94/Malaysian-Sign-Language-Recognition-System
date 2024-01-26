import cv2
import mediapipe as mp

class IProcessing:
    def init(self, cameraID):
        # define framework
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
    
    def mediapipe_process(self, handCount):
        # process image using mediapipe holistic
        results = IProcessing.holistic.process(IProcessing.Image)
        if results.face_landmarks:
            IProcessing.mpDraws.draw_landmarks(
                image=IProcessing.Image,
                landmark_list=results.face_landmarks,
                connections=IProcessing.mpFaceMesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=IProcessing.mpDrawingStyles.get_default_face_mesh_contours_style()
            )
        if handCount == 1 or handCount == 0:
            handResult = results.right_hand_landmarks
        if handCount == 2:
            handResult = results.left_hand_landmarks
        # if hand is successfully processed
        if handResult:
            # draw handlandmark on video
            x_min, x_max, y_min, y_max = IProcessing.drawBox(self, IProcessing.h, IProcessing.w, handResult)
            cropped_image = IProcessing.frame[y_min:y_max, x_min:x_max]
            cropped_image = cv2.flip(cropped_image, 1)

            cv2.rectangle(IProcessing.Image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            IProcessing.mpDraws.draw_landmarks(IProcessing.Image, handResult, IProcessing.mpHolistic.HAND_CONNECTIONS, IProcessing.mpDrawingStyles.get_default_hand_landmarks_style(), IProcessing.mpDrawingStyles.get_default_hand_connections_style())

        return IProcessing.Image
    
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
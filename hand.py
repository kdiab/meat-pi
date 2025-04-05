import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector:
    def __init__(self, display_width=1920, display_height=1080, process_width=640, process_height=360, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        
        self.process_width = process_width
        self.process_height = process_height
        
        self.display_width = display_width
        self.display_height = display_height
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # Hand landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20

    def find_hand(self, frame):
        process_frame = cv2.resize(frame, (self.process_width, self.process_height))
        
        results = self.hands.process(process_frame)
        
        hand_position = None
        display_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            scale_x = self.display_width / self.process_width
            scale_y = self.display_height / self.process_height
            
            h, w, c = display_frame.shape
            landmark_image = np.zeros((h, w, 3), np.uint8)
            
            landmark_style = self.mp_drawing_styles.DrawingSpec(color=(255, 255, 255))
            connection_style = self.mp_drawing_styles.DrawingSpec(color=(255, 255, 255))
            
            def scale_point(landmark):
                return (
                    int(landmark.x * self.process_width * scale_x),
                    int(landmark.y * self.process_height * scale_y)
                )
            
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = scale_point(hand_landmarks.landmark[start_idx])
                end_point = scale_point(hand_landmarks.landmark[end_idx])
                cv2.line(landmark_image, start_point, end_point, (255, 255, 255), 2)
            
            for landmark in hand_landmarks.landmark:
                point = scale_point(landmark)
                cv2.circle(landmark_image, point, 5, (255, 255, 255), 5)
            
            display_frame = cv2.addWeighted(display_frame, 1, landmark_image, 1, 0)
            
            wrist = hand_landmarks.landmark[self.WRIST]
            wrist_x = int(wrist.x * self.process_width * scale_x)
            wrist_y = int(wrist.y * self.process_height * scale_y)
            hand_position = (wrist_x, wrist_y)
        
        return display_frame, hand_position

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, 1)
            
            display_frame, hand_position = self.find_hand(frame)
            
            self.new_frame_time = time.time()
            fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
            self.prev_frame_time = self.new_frame_time
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if hand_position:
                x, y = hand_position
                print(f"Hand position: X={x}, Y={y}")
                cv2.putText(display_frame, f"X: {x}, Y: {y}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # ADD MIDI LOGIC HERE
            
            cv2.imshow("Hand Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandDetector(
        display_width=1920,
        display_height=1080,
        process_width=640,
        process_height=360,
        max_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector.run()

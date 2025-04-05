import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from queue import Queue

class HandDetector:
    def __init__(self, display_width=640, display_height=480, process_width=480, process_height=360, max_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3, frame_skip=2):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Try to get 30fps from camera
        
        self.process_width = process_width
        self.process_height = process_height
        
        self.display_width = display_width
        self.display_height = display_height
        self.frame_skip = frame_skip
        
        # Use lite model for better performance on Raspberry Pi
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,  
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3 
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.stopped = False
        
    def process_thread(self):
        """Thread for processing frames"""
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                process_frame = cv2.resize(frame, (self.process_width, self.process_height))
                
                process_frame = cv2.convertScaleAbs(process_frame, alpha=1.5, beta=0)
                
                process_frame = cv2.GaussianBlur(process_frame, (5, 5), 0)
                
                process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(process_frame)
                
                if not self.result_queue.full():
                    self.result_queue.put((frame, results))
            else:
                time.sleep(0.001)
    
    def find_hand(self, frame, results):
        """Process results and draw landmarks"""
        hand_position = None
        display_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * display_frame.shape[1])
                y = int(landmark.y * display_frame.shape[0])
                cv2.circle(display_frame, (x, y), 3, (255, 255, 255), 2)
            
            # Only draw a few key connections for better performance
            connections = [
                (self.WRIST, self.THUMB_TIP),
                (self.WRIST, self.INDEX_FINGER_TIP),
                (self.WRIST, self.MIDDLE_FINGER_TIP),
                (self.WRIST, self.RING_FINGER_TIP),
                (self.WRIST, self.PINKY_TIP)
            ]
            
            for start_idx, end_idx in connections:
                start_point = (
                    int(hand_landmarks.landmark[start_idx].x * display_frame.shape[1]),
                    int(hand_landmarks.landmark[start_idx].y * display_frame.shape[0])
                )
                end_point = (
                    int(hand_landmarks.landmark[end_idx].x * display_frame.shape[1]),
                    int(hand_landmarks.landmark[end_idx].y * display_frame.shape[0])
                )
                cv2.line(display_frame, start_point, end_point, (255, 255, 255), 2)
            
            # Get hand position from wrist
            wrist = hand_landmarks.landmark[self.WRIST]
            wrist_x = int(wrist.x * display_frame.shape[1])
            wrist_y = int(wrist.y * display_frame.shape[0])
            hand_position = (wrist_x, wrist_y)
        
        return display_frame, hand_position

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        processing_thread = threading.Thread(target=self.process_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        frame_counter = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, 1)
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            frame_counter += 1
            if frame_counter % self.frame_skip != 0:  
                display_frame = frame
                hand_position = None
            else:
                # Get processed result if available
                if not self.result_queue.empty():
                    processed_frame, results = self.result_queue.get()
                    display_frame, hand_position = self.find_hand(processed_frame, results)
                else:
                    display_frame = frame
                    hand_position = None
            
            # Calculate FPS
            self.new_frame_time = time.time()
            fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
            self.prev_frame_time = self.new_frame_time
            
            # Simplified text for better performance
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_position:
                x, y = hand_position
                # Print less frequently to reduce console spam
                if frame_counter % 10 == 0:
                    print(f"Hand position: X={x}, Y={y}")
                
                cv2.putText(display_frame, f"X: {x}, Y: {y}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # ADD MIDI LOGIC HERE
            
            cv2.imshow("Hand Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandDetector(
        display_width=1920,
        display_height=1080,
        process_width=480,  # Slightly higher for better distance detection
        process_height=360, # Slightly higher for better distance detection
        max_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        frame_skip=1
    )
    detector.run()

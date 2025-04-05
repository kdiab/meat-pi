import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from queue import Queue

class HandDetector:
    def __init__(self, capture_width=320, capture_height=240, display_width=640, display_height=480, max_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3, frame_skip=1):
        self.cap = cv2.VideoCapture(0)
        self.frame_skip = frame_skip
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.capture_width = capture_width
        self.capture_height = capture_height
        
        self.display_width = display_width
        self.display_height = display_height
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
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
        
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        
    def process_thread(self):
        """Thread for processing frames"""
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.5, beta=10)
                
                results = self.hands.process(rgb_frame)
                
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
            
            h, w = display_frame.shape[:2]
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(display_frame, (x, y), 3, (255, 255, 255), -1)
            
            connections = [
                (self.WRIST, self.THUMB_TIP),
                (self.WRIST, self.INDEX_FINGER_TIP),
                (self.WRIST, self.MIDDLE_FINGER_TIP),
                (self.WRIST, self.RING_FINGER_TIP),
                (self.WRIST, self.PINKY_TIP)
            ]
            
            for start_idx, end_idx in connections:
                start_point = (
                    int(hand_landmarks.landmark[start_idx].x * w),
                    int(hand_landmarks.landmark[start_idx].y * h)
                )
                end_point = (
                    int(hand_landmarks.landmark[end_idx].x * w),
                    int(hand_landmarks.landmark[end_idx].y * h)
                )
                cv2.line(display_frame, start_point, end_point, (255, 255, 255), 2)
            
            wrist = hand_landmarks.landmark[self.WRIST]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            hand_position = (wrist_x, wrist_y)
        
        display_frame = cv2.resize(display_frame, (self.display_width, self.display_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return display_frame, hand_position

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        processing_thread = threading.Thread(target=self.process_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        print("Press 'q' to quit")
        
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
                display_frame = cv2.resize(frame, (self.display_width, self.display_height), 
                                          interpolation=cv2.INTER_LINEAR)
                hand_position = None
            else:
                if not self.result_queue.empty():
                    processed_frame, results = self.result_queue.get()
                    display_frame, hand_position = self.find_hand(processed_frame, results)
                else:
                    display_frame = cv2.resize(frame, (self.display_width, self.display_height), 
                                              interpolation=cv2.INTER_LINEAR)
                    hand_position = None
            
            self.new_frame_time = time.time()
            fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
            self.prev_frame_time = self.new_frame_time
            
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hand_position:
                x, y = hand_position
                if frame_counter % 10 == 0:
                    print(f"Hand position: X={x}, Y={y}")
                
                display_x = int(x * (self.display_width / self.capture_width))
                display_y = int(y * (self.display_height / self.capture_height))
                
                cv2.putText(display_frame, f"X: {display_x}, Y: {display_y}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # ADD MIDI LOGIC HERE
            
            cv2.imshow("Hand Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandDetector(
        capture_width=320,
        capture_height=240,
        display_width=1920,
        display_height=1080,
        max_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        frame_skip = 1
    )
    detector.run()

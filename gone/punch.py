class Punch:
    def __init__(self, mp_drawing, np, cv2, deque, udp_client) -> None:
        self.np = np
      
        self.cv2 = cv2
        self.prev_pose_landmarks = None
        self.punch_history = deque(maxlen=10)
        self.drawing = mp_drawing  # Store last 10 frames for smoothing
        self.punch_threshold = 0.25  # Distance threshold for punch detection
        self.min_punch_velocity = 0.1# Reduced for better detection
        self.udp_client = udp_client
        self.fist_threshold = 0.15

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        distance = self.np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
        return distance

    def calculate_velocity(self, prev_pos, curr_pos):
        """Calculate velocity of hand movement"""
        if prev_pos is None:
            return 0
        return self.calculate_distance(prev_pos, curr_pos)
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points (in 3D space)
        Returns angle in degrees
        point1: start point
        point2: vertex (center point)
        point3: end point
        """
        # Create vectors
        v1 = self.np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
        v2 = self.np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])

        # Calculate magnitudes
        mag1 = self.np.linalg.norm(v1)
        mag2 = self.np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            return 0

        # Calculate cosine of angle
        cos_angle = self.np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = self.np.clip(cos_angle, -1.0, 1.0)  # Clamp to [-1, 1]

        # Calculate angle in degrees
        angle = self.np.arccos(cos_angle) * 180 / self.np.pi
        return angle
  
 
    def is_fist_closed(self, hand_landmarks):
        """
        Check if hand is closed (fist) by measuring distances between fingertips and palm
        Returns True if fist is closed
        """
        if hand_landmarks is None:
            return False
        
        try:
            # MediaPipe hand landmark indices
            WRIST = 0
            THUMB_TIP = 4
            INDEX_TIP = 8
            MIDDLE_TIP = 12
            RING_TIP = 16
            PINKY_TIP = 20
            
            wrist = hand_landmarks.landmark[WRIST]
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            index_tip = hand_landmarks.landmark[INDEX_TIP]
            middle_tip = hand_landmarks.landmark[MIDDLE_TIP]
            ring_tip = hand_landmarks.landmark[RING_TIP]
            pinky_tip = hand_landmarks.landmark[PINKY_TIP]
            
            # Calculate average distance of fingertips to wrist
            distances = [
                self.calculate_distance(index_tip, wrist),
                self.calculate_distance(middle_tip, wrist),
                self.calculate_distance(ring_tip, wrist),
                self.calculate_distance(pinky_tip, wrist)
            ]
            
            avg_distance = sum(distances) / len(distances)
            
            # If fingertips are close to wrist, hand is closed (fist)
            is_closed = avg_distance < self.fist_threshold
            
            return is_closed
            
        except Exception as e:
            print(f"Error checking fist: {e}")
            return False
  
  
  
  
  
             
    def draw_punch_indicator(self, frame, punch_detected, confidence, arm_angle, punch_type, distance):
        """Draw punch detection indicator on frame"""
        if frame is None:
            return frame

        try:
            h, w, c = frame.shape
        except:
            h, w = frame.shape[:2]

        if punch_detected:
            # Draw red border for punch
            self.cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)

            # Draw punch text
            text = f"PUNCH DETECTED! {punch_type} - Confidence: {confidence:.2f}"
            self.cv2.putText(frame, text, (60, 60), self.cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 0, 255), 3)

            # Draw angle and distance info
            angle_text = f"Arm Angle: {arm_angle:.1f}°"
            self.cv2.putText(frame, angle_text, (100, 100), self.cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

            dist_text = f"Distance: {distance:.2f}"
            self.cv2.putText(frame, dist_text, (200, 200), self.cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

            # Draw filled circle indicator
            self.cv2.circle(frame, (400, 400), 30, (0, 0, 255), -1)
        else:
            # Draw status text
            self.cv2.putText(frame, "No Punch Detected", (50, 50), self.cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

        return frame

    def detect_punch(self, landmarks, prev_landmarks,left_hand,right_hand):
        """
        Detect punch based on:
        1. Rapid hand movement (velocity)
        2. Forward motion (z-coordinate)
        3. Arm extension (distance from shoulder)
        4. Works with both straight and bent arms
        """
        if landmarks is None or prev_landmarks is None:
            return False, 0, "", 0, 0

        try:
            # MediaPipe landmarks indices
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14

            # Get current positions
            left_wrist = landmarks.landmark[LEFT_WRIST]
            right_wrist = landmarks.landmark[RIGHT_WRIST]
            left_shoulder = landmarks.landmark[LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[RIGHT_SHOULDER]
            left_elbow = landmarks.landmark[LEFT_ELBOW]
            right_elbow = landmarks.landmark[RIGHT_ELBOW]

            # Get previous positions
            prev_left_wrist = prev_landmarks.landmark[LEFT_WRIST]
            prev_right_wrist = prev_landmarks.landmark[RIGHT_WRIST]

            # Calculate velocities
            left_velocity = self.calculate_distance(prev_left_wrist, left_wrist)
            right_velocity = self.calculate_distance(prev_right_wrist, right_wrist)

            punch_detected = False
            punch_type = ""
            confidence = 0
            arm_angle = 0
            distance = 0

            right_fist_closed = self.is_fist_closed(right_hand)
            left_fist_closed = self.is_fist_closed(left_hand)
            # Right punch detection
            if right_velocity > self.min_punch_velocity and right_fist_closed:
                # Calculate arm angle (shoulder -> elbow -> wrist)
                arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Calculate distance between wrist and shoulder
                distance = self.calculate_distance(right_wrist, right_shoulder)
                
                # Check forward motion


                # Accept angles from 90° to 180° (bent to straight arm)
                # and check if arm is extended forward
                if arm_angle >= 90 and distance > self.punch_threshold  :
                    print(f"Right punch - Velocity: {right_velocity:.3f}, Angle: {arm_angle:.1f}°, Distance: {distance:.3f}")
                    punch_detected = True
                    punch_type = "RIGHT"
                    
                    # Confidence based on velocity and extension
                    velocity_conf = min(right_velocity / 0.3, 1.0)
                    angle_conf = min((arm_angle - 90) / 90, 1.0)
                    confidence = (velocity_conf + angle_conf) / 2
                    confidence = max(0, min(confidence, 1.0))

            # Left punch detection
            if left_velocity > self.min_punch_velocity and left_fist_closed:
                # Calculate arm angle (shoulder -> elbow -> wrist)
                arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Calculate distance between wrist and shoulder
                distance = self.calculate_distance(left_wrist, left_shoulder)
                
                # Check forward motion


                # Accept angles from 90° to 180° (bent to straight arm)
                # and check if arm is extended forward
                if arm_angle >= 90 and distance > self.punch_threshold :
                    print(f"Left punch - Velocity: {left_velocity:.3f}, Angle: {arm_angle:.1f}°, Distance: {distance:.3f}")
                    punch_detected = True
                    punch_type = "LEFT"
                    
                    # Confidence based on velocity and extension
                    velocity_conf = min(left_velocity / 0.3, 1.0)
                    angle_conf = min((arm_angle - 90) / 90, 1.0)
                    confidence = (velocity_conf + angle_conf) / 2
                    confidence = max(0, min(confidence, 1.0))

            return punch_detected, confidence, punch_type, arm_angle, distance

        except Exception as e:
            print(f"Error in punch detection: {e}")
            return False, 0, "", 0, 0
    
    def punch_execute(self, frame, result,hand_results):
        # Detect punch
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        if hand_results and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                if hand_label == "Left":
                    left_hand_landmarks = hand_landmarks
                elif hand_label == "Right":
                    right_hand_landmarks = hand_landmarks
        

        punch_detected, confidence, punch_type, arm_angle, distance = self.detect_punch(
            result.pose_landmarks,
            self.prev_pose_landmarks,left_hand_landmarks,right_hand_landmarks)
        
        # Store in history for smoothing
        self.punch_history.append(punch_detected)
       
        # if 6+ frames in last 10 show punch
        smoothed_punch = sum(self.punch_history) >= 6
 
        draw_frame = self.draw_punch_indicator(frame, smoothed_punch, confidence, arm_angle, punch_type, distance)
        self.prev_pose_landmarks = result.pose_landmarks
        
        if self.udp_client:
            self.udp_client.send_punch_data(
                smoothed_punch, 
                confidence, 
                punch_type, 
                arm_angle, 
                distance)
        
        return draw_frame
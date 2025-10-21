class Fire:
    def __init__(self, mp_hands, mp_pose, mp_drawing, np, cv2):
        self.mp_hands = mp_hands
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.np = np
        self.cv2 = cv2
    
    def is_hand_open(self, hand_landmarks):
        """Check if hand is open by comparing finger tip positions with knuckles"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        open_fingers = 0
        
        # Check each finger (index to pinky)
        for i in range(1, 5):
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]
            
            if tip.y < pip.y:  # Tip is above PIP joint (extended)
                open_fingers += 1
        
        # Check thumb separately (horizontal extension)
        thumb_tip = hand_landmarks.landmark[finger_tips[0]]
        thumb_ip = hand_landmarks.landmark[finger_pips[0]]
        
        if abs(thumb_tip.x - thumb_ip.x) > 0.04:  # Thumb extended
            open_fingers += 1
        
        return open_fingers >= 4  # At least 4 fingers must be extended
    
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
    
    def is_arm_raised_at_shoulder(self, pose_landmarks, side='left'):
        """Check if arm is raised up at shoulder level (vertical position)"""
        if side == 'left':
            shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        else:
            shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check if arm is raised up (wrist should be above shoulder)
        is_raised = wrist.y < shoulder.y  # Lower y value means higher position
        
        # Check if arm is roughly vertical (x-coordinates similar)
        x_diff_shoulder_wrist = abs(shoulder.x - wrist.x)
        is_vertical = x_diff_shoulder_wrist < 0.15  # Arm stays close to body vertically
        
        # Check if arm is straight (angle at elbow close to 180 degrees)
        angle = self.calculate_angle(shoulder, elbow, wrist)
        is_straight = angle >= 90  # More strict for vertical arms
        
        return is_raised and is_vertical and is_straight
    
    def are_both_arms_raised(self, pose_landmarks):
        """Check if both arms are raised at shoulder height"""
        left_raised = self.is_arm_raised_at_shoulder(pose_landmarks, side='left')
        right_raised = self.is_arm_raised_at_shoulder(pose_landmarks, side='right')
        return left_raised and right_raised
    
    def Detect_Fire(self, frame, pose_results, hands_results):
        """
        Main detection method for fire pose
        Returns: (frame, fire_pose_detected)
        """
        fire_pose_detected = False
        status_text = []
        left_hand_open = False
        right_hand_open = False
        left_arm_straight = False
        right_arm_straight = False
        
        # Check hands
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                   hands_results.multi_handedness):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                hand_label = handedness.classification[0].label
                is_open = self.is_hand_open(hand_landmarks)
                
                if hand_label == 'Left':
                    left_hand_open = is_open
                else:
                    right_hand_open = is_open
        
        # Check pose
        if pose_results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            left_arm_straight = self.is_arm_raised_at_shoulder(
                pose_results.pose_landmarks, 'left')
            right_arm_straight = self.is_arm_raised_at_shoulder(
                pose_results.pose_landmarks, 'right')
        
        # Update status
        status_text.append(f"Left Hand: {'OPEN' if left_hand_open else 'closed'}")
        status_text.append(f"Right Hand: {'OPEN' if right_hand_open else 'closed'}")
        status_text.append(f"Left Arm: {'STRAIGHT' if left_arm_straight else 'bent'}")
        status_text.append(f"Right Arm: {'STRAIGHT' if right_arm_straight else 'bent'}")
        
        # Check if fire pose is detected
        if (left_hand_open and right_hand_open and 
            left_arm_straight and right_arm_straight):
            fire_pose_detected = True
        
        # Display status on frame
        y_offset = 30
        for text in status_text:
            self.cv2.putText(frame, text, (10, y_offset), 
                       self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Display fire pose detection
        if fire_pose_detected:
            self.cv2.putText(frame, "FIRE POSE DETECTED!", (10, y_offset + 20), 
                       self.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Add visual effect
            self.cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                         (0, 0, 255), 10)
        return frame, fire_pose_detected
        
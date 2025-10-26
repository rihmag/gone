from RotationUDP import RotationUDPClient
class Rotation:
    def  __init__(self,mp_pose,mp_drawing,cv2,math,udp_host,udp_port) -> None:
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.cv2  = cv2
        self.math = math
        self.reference_angle = None
        self.calibration_frames = 30
        self.frame_count = 0
        self.udp_client = RotationUDPClient(host=udp_host, port=udp_port)
        
# Initialize MediaPipe Pose





    def calculate_angle(self,point1, point2):
        """Calculate angle between two points relative to horizontal axis"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = self.math.degrees(self.math.atan2(dy, dx))
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180

        return angle

    def get_rotation_direction_and_angle(self,current_angle, reference_angle=0):
        """
        Determine rotation direction and angle from reference
        Reference angle of 0 means shoulders are level (horizontal)
        Positive angle = rotated right (right shoulder up)
        Negative angle = rotated left (left shoulder up)
        """
        rotation = current_angle - reference_angle

        if rotation > 90:
            rotation = rotation - 180
        elif rotation < -90:
            rotation = rotation + 180

        if abs(rotation) < 7:  # Small dead zone for noise only
            direction = "Center"
        elif rotation > 0:
            direction = "Rotated Right"
        else:
            direction = "Rotated Left"
        return direction , rotation 

    def draw_info(self,frame, direction, angle, left_shoulder, right_shoulder):
        """Draw information on frame"""
        

        # Draw shoulder line
        self.cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)

        # Draw shoulder points
        self.cv2.circle(frame, left_shoulder, 8, (255, 0, 0), -1)
        self.cv2.circle(frame, right_shoulder, 8, (0, 0, 255), -1)

        # Display info
        info_text = f"Direction: {direction}"
        angle_text = f"Rotation Angle: {angle:.1f} degrees"

        self.cv2.putText(frame, info_text, (10, 30), 
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self.cv2.putText(frame, angle_text, (10, 70), 
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Labels for shoulders
        self.cv2.putText(frame, "L", (left_shoulder[0] - 20, left_shoulder[1] - 10), 
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        self.cv2.putText(frame, "R", (right_shoulder[0] + 10, right_shoulder[1] - 10), 
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def execute(self,frame,results):
        direction = None
        rotation_angle = None
        h,w, _ = frame.shape

        # Open webcam
        

        # Reference angle (calibration)
    
    
    

        print("Starting shoulder rotation detector...")
        print("Stand straight facing the camera for calibration (first 30 frames)")
        print("Press 'r' to recalibrate")
        print("Press 'q' to quit")


            # Convert to RGB
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get shoulder landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Convert to pixel coordinates
            left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))

            # Calculate current shoulder angle
            current_angle = self.calculate_angle(left_shoulder_coords, right_shoulder_coords)

            # Calibration phase
            if self.frame_count < self.calibration_frames:
                if self.reference_angle is None:
                    self.reference_angle = current_angle
                else:
                    # Average for better calibration
                    self.reference_angle = (self.reference_angle * self.frame_count + current_angle) / (self.frame_count + 1)
                self.frame_count += 1

                self.cv2.putText(frame, f"Calibrating... {self.frame_count}/{self.calibration_frames}", 
                            (10, h - 20), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.udp_client.send_rotation_data("Calibrating", 0.0, is_calibrated=False)
            else:
                # Detection phase
                direction, rotation_angle = self.get_rotation_direction_and_angle(current_angle, self.reference_angle)
                self.draw_info(frame, direction, rotation_angle, left_shoulder_coords, right_shoulder_coords)
                self.udp_client.send_rotation_data(direction, rotation_angle, is_calibrated=True)
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
                
        else:
            self.cv2.putText(frame, "No person detected", (10, 30), 
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.udp_client.send_rotation_data("No Detection", 0.0, is_calibrated=False)
            # Display frame

        return frame , direction , rotation_angle
            # Keyboard contro
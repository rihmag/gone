import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class Walk:
    def __init__(self, velocity_threshold=0.015, sensitivity=0.8, history_size=30, min_swings=3, cycle_gap_threshold=1.0):
        """
        Initialize the shoulder shake detector.
        
        Parameters:
        - velocity_threshold: Minimum velocity for shake detection (0.001-0.1, default 0.015)
        - sensitivity: Detection sensitivity (0.1-1.0, default 0.8)
        - history_size: Number of frames to track for smoothing (default 30)
        - min_swings: Minimum alternating swings to count as one cycle (default 3)
        - cycle_gap_threshold: Max time gap between cycles to keep "ongoing" status (default 1.0 seconds)
        """
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configurable parameters
        self.velocity_threshold = velocity_threshold
        self.sensitivity = sensitivity
        self.history_size = history_size
        self.min_swings = min_swings  # Minimum swings to complete a cycle
        self.cycle_gap_threshold = cycle_gap_threshold  # Max gap between cycles
        
        # Tracking variables
        self.left_shoulder_history = deque(maxlen=history_size)
        self.right_shoulder_history = deque(maxlen=history_size)
        self.left_velocities = deque(maxlen=history_size)
        self.right_velocities = deque(maxlen=history_size)
        
        # Cycle tracking
        self.shake_cycles = 0
        self.current_swing_count = 0  # Count swings in current cycle
        self.last_active_shoulder = None  # Track which shoulder moved last
        self.cycle_in_progress = False
        self.last_swing_time = time.time()
        self.cycle_start_time = None
        self.last_cycle_completion_time = None  # Track when last cycle was completed
        self.cycle_session_active = False  # True if cycles are happening continuously
        
        # Visualization
        self.cycle_display_time = 0
        self.swing_times = []  # Track timing of swings for frequency calculation
        
    def calculate_velocity(self, current_pos, history):
        """Calculate the velocity of shoulder movement."""
        if len(history) < 2:
            return 0
        
        prev_pos = history[-1]
        velocity = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
        return velocity
    
    def detect_peak(self, velocities, threshold):
        """Detect if there's a peak in velocity indicating a shake."""
        if len(velocities) < 5:
            return False
        
        recent_velocities = list(velocities)[-5:]
        avg_velocity = np.mean(recent_velocities)
        
        # Check if current velocity exceeds threshold
        return avg_velocity > threshold * self.sensitivity
    
    def update_cycle_detection(self, left_peak, right_peak):
        """Update the shake cycle detection logic for continuous swinging."""
        current_time = time.time()
        cycle_completed = False
        
        # Detect alternating shoulder movements
        swing_detected = False
        
        if left_peak and self.last_active_shoulder != 'left':
            # Left shoulder swing detected
            swing_detected = True
            self.last_active_shoulder = 'left'
            
            if not self.cycle_in_progress:
                # Start new cycle
                self.cycle_in_progress = True
                self.cycle_start_time = current_time
                self.current_swing_count = 1
                
                # Check if this is part of an ongoing session
                if self.last_cycle_completion_time is not None:
                    time_since_last_cycle = current_time - self.last_cycle_completion_time
                    if time_since_last_cycle <= self.cycle_gap_threshold:
                        self.cycle_session_active = True
                    else:
                        self.cycle_session_active = False
            else:
                # Continue cycle
                self.current_swing_count += 1
            
            self.swing_times.append(current_time)
            if len(self.swing_times) > 10:
                self.swing_times.pop(0)
        
        elif right_peak and self.last_active_shoulder != 'right':
            # Right shoulder swing detected
            swing_detected = True
            self.last_active_shoulder = 'right'
            
            if not self.cycle_in_progress:
                # Start new cycle
                self.cycle_in_progress = True
                self.cycle_start_time = current_time
                self.current_swing_count = 1
                
                # Check if this is part of an ongoing session
                if self.last_cycle_completion_time is not None:
                    time_since_last_cycle = current_time - self.last_cycle_completion_time
                    if time_since_last_cycle <= self.cycle_gap_threshold:
                        self.cycle_session_active = True
                    else:
                        self.cycle_session_active = False
            else:
                # Continue cycle
                self.current_swing_count += 1
            
            self.swing_times.append(current_time)
            if len(self.swing_times) > 10:
                self.swing_times.pop(0)
        
        # Update last swing time if any swing detected
        if swing_detected:
            self.last_swing_time = current_time
        
        # Check if cycle is complete (enough continuous swings)
        if self.cycle_in_progress and self.current_swing_count >= self.min_swings * 2:
            # Complete the cycle
            self.shake_cycles += 1
            self.cycle_display_time = current_time
            self.last_cycle_completion_time = current_time
            cycle_completed = True
            
            # Mark session as active since we just completed a cycle
            self.cycle_session_active = True
            
            # Reset for next cycle but keep momentum
            self.current_swing_count = 0
            self.cycle_start_time = current_time
        
        # Check if session should end (no cycle completion within gap threshold)
        if self.last_cycle_completion_time is not None:
            time_since_last_completion = current_time - self.last_cycle_completion_time
            if time_since_last_completion > self.cycle_gap_threshold:
                self.cycle_session_active = False
        
        # Reset if swinging stops (no movement for 1 second)
        if self.cycle_in_progress and (current_time - self.last_swing_time) > 2:
            self.cycle_in_progress = False
            self.current_swing_count = 0
            self.last_active_shoulder = None
            self.cycle_start_time = None
        
        return cycle_completed
    
    def get_swing_frequency(self):
        """Calculate current swing frequency in Hz."""
        if len(self.swing_times) < 2:
            return 0.0
        
        time_span = self.swing_times[-1] - self.swing_times[0]
        if time_span == 0:
            return 0.0
        
        return (len(self.swing_times) - 1) / time_span
    
    def process_frame(self, frame):
        """Process a single frame and detect shoulder shakes."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        cycle_completed = False
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Get shoulder positions (landmark indices: 11=left, 12=right)
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            left_pos = (left_shoulder.x, left_shoulder.y)
            right_pos = (right_shoulder.x, right_shoulder.y)
            
            # Calculate velocities
            left_vel = self.calculate_velocity(left_pos, self.left_shoulder_history)
            right_vel = self.calculate_velocity(right_pos, self.right_shoulder_history)
            
            # Update histories
            self.left_shoulder_history.append(left_pos)
            self.right_shoulder_history.append(right_pos)
            self.left_velocities.append(left_vel)
            self.right_velocities.append(right_vel)
            
            # Detect peaks
            left_peak = self.detect_peak(self.left_velocities, self.velocity_threshold)
            right_peak = self.detect_peak(self.right_velocities, self.velocity_threshold)
            
            # Update cycle detection
            cycle_completed = self.update_cycle_detection(left_peak, right_peak)
            
            # Visualizations
            h, w = frame.shape[:2]
            
            # Draw shoulder indicators
            left_x, left_y = int(left_pos[0] * w), int(left_pos[1] * h)
            right_x, right_y = int(right_pos[0] * w), int(right_pos[1] * h)
            
            # Color shoulders based on detection and which was last active
            left_color = (0, 255, 0) if (left_peak or self.last_active_shoulder == 'left') else (0, 165, 255)
            right_color = (0, 255, 0) if (right_peak or self.last_active_shoulder == 'right') else (0, 165, 255)
            
            cv2.circle(frame, (left_x, left_y), 15, left_color, -1)
            cv2.circle(frame, (right_x, right_y), 15, right_color, -1)
            
            # Draw velocity bars
            left_bar_height = int(min(left_vel / self.velocity_threshold * 100, 200))
            right_bar_height = int(min(right_vel / self.velocity_threshold * 100, 200))
            
            cv2.rectangle(frame, (50, h - 50), (70, h - 50 - left_bar_height), (255, 0, 0), -1)
            cv2.rectangle(frame, (w - 70, h - 50), (w - 50, h - 50 - right_bar_height), (255, 0, 0), -1)
            
            cv2.putText(frame, "L", (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "R", (w - 70, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display statistics
        cv2.putText(frame, f"Completed Cycles: {self.shake_cycles}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display "Cycle is Going On" status
        if self.cycle_session_active:
            session_status = "CYCLE IS GOING ON: TRUE"
            session_color = (0, 255, 0)  # Green
            
            # Calculate time since last cycle completion
            if self.last_cycle_completion_time is not None:
                time_since = time.time() - self.last_cycle_completion_time
                time_remaining = self.cycle_gap_threshold - time_since
                if time_remaining > 0:
                    session_status += f" ({time_remaining:.1f}s)"
        else:
            session_status = "CYCLE IS GOING ON: FALSE"
            session_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, session_status, (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, session_color, 2)
        
        # Show current swing count and progress
        if self.cycle_in_progress:
            progress = f"Swings: {self.current_swing_count}/{self.min_swings * 2}"
            cv2.putText(frame, progress, (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Progress bar
            bar_width = 200
            bar_x = 10
            bar_y = 130
            progress_pct = min(self.current_swing_count / (self.min_swings * 2), 1.0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress_pct), bar_y + 20), (0, 255, 0), -1)
        
        # Show swing frequency
        freq = self.get_swing_frequency()
        cv2.putText(frame, f"Frequency: {freq:.1f} Hz", (10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Velocity Threshold: {self.velocity_threshold:.3f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sensitivity: {self.sensitivity:.1f}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show cycle gap threshold
        cv2.putText(frame, f"Cycle Gap: {self.cycle_gap_threshold:.1f}s", (10, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Flash "CYCLE COMPLETE!" when a cycle is detected
        if time.time() - self.cycle_display_time < 0.5:
            cv2.putText(frame, "CYCLE COMPLETE!", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Show cycle progress
        if self.cycle_in_progress:
            next_shoulder = "RIGHT" if self.last_active_shoulder == 'left' else "LEFT"
            status = f"SWINGING! Next: {next_shoulder}"
            color = (0, 255, 255)
        else:
            status = "Start continuous shoulder swings (L-R-L-R...)"
            color = (200, 200, 200)
        
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, cycle_completed
    
    def run(self):
        """Run the shoulder shake detector."""
        cap = cv2.VideoCapture(0)
        
        print("Shoulder Shake Detector - Continuous Swing Mode")
        print("==============================================")
        print("Controls:")
        print("  q - Quit")
        print("  r - Reset cycle count")
        print("  + - Increase velocity threshold")
        print("  - - Decrease velocity threshold")
        print("  ] - Increase sensitivity")
        print("  [ - Decrease sensitivity")
        print("  > - Increase minimum swings per cycle")
        print("  < - Decrease minimum swings per cycle")
        print("  } - Increase cycle gap threshold")
        print("  { - Decrease cycle gap threshold")
        print("\nSwing your shoulders continuously (L-R-L-R-L-R...) to complete cycles!")
        print(f"Current setting: {self.min_swings * 2} alternating swings = 1 cycle")
        print(f"Cycle gap threshold: {self.cycle_gap_threshold}s (max time between cycles)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, cycle_completed = self.process_frame(frame)
            
            # Display
            cv2.imshow('Shoulder Shake Detector', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.shake_cycles = 0
                self.current_swing_count = 0
                self.cycle_in_progress = False
                self.cycle_session_active = False
                self.last_cycle_completion_time = None
                print("Cycle count and session reset!")
            elif key == ord('+') or key == ord('='):
                self.velocity_threshold = min(self.velocity_threshold + 0.001, 0.1)
                print(f"Velocity threshold: {self.velocity_threshold:.3f}")
            elif key == ord('-'):
                self.velocity_threshold = max(self.velocity_threshold - 0.001, 0.001)
                print(f"Velocity threshold: {self.velocity_threshold:.3f}")
            elif key == ord(']'):
                self.sensitivity = min(self.sensitivity + 0.1, 1.0)
                print(f"Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('['):
                self.sensitivity = max(self.sensitivity - 0.1, 0.1)
                print(f"Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('>') or key == ord('.'):
                self.min_swings = min(self.min_swings + 1, 10)
                print(f"Minimum swings per cycle: {self.min_swings * 2}")
            elif key == ord('<') or key == ord(','):
                self.min_swings = max(self.min_swings - 1, 1)
                print(f"Minimum swings per cycle: {self.min_swings * 2}")
            elif key == ord('}') or key == ord(']'):
                self.cycle_gap_threshold = min(self.cycle_gap_threshold + 0.1, 5.0)
                print(f"Cycle gap threshold: {self.cycle_gap_threshold:.1f}s")
            elif key == ord('{') or key == ord('['):
                self.cycle_gap_threshold = max(self.cycle_gap_threshold - 0.1, 0.5)
                print(f"Cycle gap threshold: {self.cycle_gap_threshold:.1f}s")
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()


if __name__ == "__main__":
    # Initialize detector with custom parameters
    detector = Walk(
        velocity_threshold=0.011,  # Adjust for faster/slower shake detection
        sensitivity=1.2,           # Higher = more sensitive to movements
        history_size=30,           # Smoothing window
        min_swings=3               # Number of alternating swings to complete a cycle (3 = 6 total swings)
    )
    
    # Run the detector
    detector.run()
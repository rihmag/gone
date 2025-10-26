import socket
import json
class PunchUDPClient:
    def __init__(self, host='127.0.0.1', port=5065):
        """Initialize UDP client to send punch data to Unity"""
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP Client initialized - Sending to {host}:{port}")
    
    def send_punch_data(self, punch_detected, confidence, punch_type, arm_angle, distance):
        """
        Send punch detection data to Unity
        
        Args:
            punch_detected (bool): Whether punch is detected
            confidence (float): Confidence score (0-1)
            punch_type (str): "LEFT" or "RIGHT"
            arm_angle (float): Arm extension angle
            distance (float): Wrist-shoulder distance
        """
        try:
            # Create data dictionary
            data = {
                "punch_detected": punch_detected,
                "confidence": round(confidence, 3),
                "punch_type": punch_type,
                "arm_angle": round(arm_angle, 2),
                "distance": round(distance, 3)
            }
            
            # Convert to JSON string
            json_data = json.dumps(data)
            
            # Send via UDP
            self.sock.sendto(json_data.encode('utf-8'), (self.host, self.port))
            
        except Exception as e:
            print(f"Error sending UDP data: {e}")
    
    def close(self):
        """Close the socket"""
        self.sock.close()
        print("UDP Client closed")

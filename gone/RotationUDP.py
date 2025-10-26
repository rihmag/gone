import socket
import json
class RotationUDPClient:
     def __init__(self, host='127.0.0.1', port=5066):
        """Initialize UDP client to send rotation data to Unity"""
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Rotation UDP Client initialized - Sending to {host}:{port}")
    
     def send_rotation_data(self, direction, rotation_angle, is_calibrated=True):
        """
        Send shoulder rotation data to Unity
        
        Args:
            direction (str): Rotation direction ("Center", "Rotated Right", "Rotated Left")
            rotation_angle (float): Rotation angle in degrees
            is_calibrated (bool): Whether calibration is complete
        """
        try:
            # Create data dictionary
            data = {
                "direction": direction,
                "rotation_angle": round(rotation_angle, 2),
                "is_calibrated": is_calibrated,
                "abs_angle": round(abs(rotation_angle), 2)
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
        print("Rotation UDP Client closed")
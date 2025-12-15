import socket
import json
import numpy as np

class WalkUDPClient:
    def __init__(self, host='127.0.0.1', port=5067):
        """Initialize UDP client to send walk data to Unity"""
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Walk UDP Client initialized - Sending to {host}:{port}")
    
    def _convert_to_native(self, value):
        """Convert numpy types to native Python types"""
        if isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.item()
        else:
            return value
    
    def send_walk_data(self, cycle_session_active, cycle_in_progress, 
                       last_active_shoulder, left_peak, right_peak):
        """
        Send walk cycle data to Unity
        """
        try:
            # Convert all values to native Python types
            data = {
                "cycle_session_active": bool(self._convert_to_native(cycle_session_active)),
                "cycle_in_progress": bool(self._convert_to_native(cycle_in_progress)),
                "last_active_shoulder": str(last_active_shoulder),
                "left_peak": round(float(self._convert_to_native(left_peak)), 2),
                "right_peak": round(float(self._convert_to_native(right_peak)), 2)
            }
            
            # Convert to JSON string
            json_data = json.dumps(data)
            
            # Send via UDP
            self.sock.sendto(json_data.encode('utf-8'), (self.host, self.port))
            
        except Exception as e:
            print(f"Error sending walk UDP data: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Close the socket"""
        self.sock.close()
        print("Walk UDP Client closed")
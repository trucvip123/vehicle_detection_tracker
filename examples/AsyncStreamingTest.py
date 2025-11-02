"""
Example: Async streaming camera with license plate display
For high-performance streaming scenarios
"""

import asyncio
import cv2
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

async def main():
    tracker = VehicleDetectionTracker()
    
    try:
        print("Starting async camera stream... (Press 'q' to quit)")
        
        async for frame in tracker.process_camera_stream_async(camera_index=0):
            cv2.imshow("Vehicle Detection - Async Streaming", frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    finally:
        tracker.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())


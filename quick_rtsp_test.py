"""Quick RTSP Test - No GUI (Headless)"""
import cv2
import time

RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"

print("🔌 Testing RTSP connection...")
print(f"URL: {RTSP_URL}\n")

# Try FFMPEG backend
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if cap.isOpened():
    print("✅ RTSP connection established!")
    
    frame_count = 0
    start = time.time()
    timeout = 10  # seconds
    
    print(f"📹 Reading frames for {timeout} seconds...\n")
    
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            elapsed = time.time() - start
            fps = frame_count / elapsed
            print(f"  Frame {frame_count}: {frame.shape} | FPS: {fps:.1f}", end='\r')
        time.sleep(0.01)
    
    cap.release()
    
    print(f"\n\n✅ TEST RESULTS:")
    print(f"  Frames captured: {frame_count}")
    print(f"  Duration: {time.time() - start:.1f}s")
    if frame_count > 0:
        print(f"  Average FPS: {frame_count / (time.time() - start):.1f}")
        print(f"\n🟢 RTSP PIPELINE WORKING SUCCESSFULLY!")
    else:
        print(f"\n⚠️  Connected but no frames received")
        
else:
    print("❌ Failed to connect to RTSP stream")
    print("\nTroubleshooting:")
    print("  1. Verify URL is reachable")
    print("  2. Check camera is online")
    print("  3. Test with VLC: Media → Open Network Stream")

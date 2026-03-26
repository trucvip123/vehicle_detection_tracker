import cv2
import time


RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"


def create_capture_ffmpeg(rtsp_url: str):
    """Your current method (FFmpeg)"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def create_capture_gstreamer(rtsp_url: str):
    """Optimized GStreamer + GPU decode"""
    pipeline = (
        f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! "
        f"rtph264depay ! h264parse ! nvh264dec ! "
        f"videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def test_capture(cap):
    if not cap.isOpened():
        print("❌ Cannot open stream")
        return

    print("✅ Stream opened")

    frame_count = 0
    start_time = time.time()
    last_time = start_time

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Lost frame")
            continue

        frame_count += 1

        # FPS calculation
        now = time.time()
        fps = frame_count / (now - start_time)

        # Instant FPS
        inst_fps = 1 / (now - last_time) if now != last_time else 0
        last_time = now

        cv2.putText(frame, f"FPS(avg): {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS(inst): {inst_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("RTSP Test", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose mode:")
    print("1 - FFmpeg (current)")
    print("2 - GStreamer (GPU)")

    mode = input("Select: ")

    if mode == "1":
        cap = create_capture_ffmpeg(RTSP_URL)
    else:
        cap = create_capture_gstreamer(RTSP_URL)

    test_capture(cap)
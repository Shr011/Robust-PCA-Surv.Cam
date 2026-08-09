# test_camera.py
import cv2
from config import PHONE_STREAM_URL

cap = cv2.VideoCapture(PHONE_STREAM_URL)

if not cap.isOpened():
    print("❌ Cannot connect to phone camera.")
    print("   → Check if phone and laptop are on same Wi-Fi")
    print("   → Check if IP Webcam server is running on phone")
else:
    print("✅ Connected to phone camera!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Phone Camera Feed - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
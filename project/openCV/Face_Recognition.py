import cv2
import time
import os

face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

os.makedirs('screenshots', exist_ok=True)

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Press 's' to save screenshot | Press 'q' to quit")

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)

    cv2.putText(canvas, "Press 'S' to Screenshot | 'Q' to Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Face Detection', canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f'screenshots/capture_{time.strftime("%d_%B")}.jpg'
        cv2.imwrite(filename, canvas)
        print(f"📸 Screenshot saved: {filename}")

        cv2.putText(canvas, "Screenshot Saved!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Face Detection', canvas)
        cv2.waitKey(500)  # Show message for 0.5 seconds

    if key == ord('q') or key == 27:
        break

    if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()
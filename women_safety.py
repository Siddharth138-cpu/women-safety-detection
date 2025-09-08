import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("D:\\projects\\ddddd\\yolov8n (1).pt")

# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe('D:\\projects\\ddddd\\deploy_gender.prototxt', 'D:\\projects\\ddddd\\gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']

# Load Haar face detector
face_cascade = cv2.CascadeClassifier("C:\\Users\\win 10\\Downloads\\haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set fullscreen window
cv2.namedWindow("Women Safety Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Women Safety Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Functions
def get_danger_distance(frame_width):
    return frame_width // 4  # 1280 / 4 = 320 px

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def classify_gender(face_crop):
    try:
        if face_crop.size == 0:
            return "Unknown"
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                     (78.426, 87.768, 114.895), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        return GENDER_LIST[gender_preds[0].argmax()]
    except Exception as e:
        print("Gender classification error:", e)
        return "Unknown"

def boxes_overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    overlap_area = overlap_x * overlap_y

    return overlap_area > 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    danger_distance = get_danger_distance(frame.shape[1])

    results = model(frame)[0]
    person_boxes = []
    genders = []

    # Detect people and classify gender
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = result
        if int(cls) == 0 and score > 0.5:
            box = [int(x1), int(y1), int(x2), int(y2)]
            person_boxes.append(box)

            person_roi = frame[box[1]:box[3], box[0]:box[2]]
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                face_crop = person_roi[fy:fy+fh, fx:fx+fw]
                gender = classify_gender(face_crop)
            else:
                gender = "Unknown"

            genders.append(gender)

            # Draw bounding box
            color = (0, 255, 0) if gender == 'Female' else (255, 0, 0) if gender == 'Male' else (128, 128, 128)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, gender, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Separate boxes
    female_boxes = [box for box, gender in zip(person_boxes, genders) if gender == 'Female']
    male_boxes = [box for box, gender in zip(person_boxes, genders) if gender == 'Male']

    threat_detected = False

    for f_box in female_boxes:
        f_center = get_center(f_box)
        cv2.rectangle(frame, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (255, 0, 255), 3)

        for m_box in male_boxes:
            m_center = get_center(m_box)
            dist = euclidean_distance(f_center, m_center)

            # Show distance
            cv2.putText(frame, f"{int(dist)} px", (m_center[0], m_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(f"Distance between male and female: {int(dist)} | Danger threshold: {danger_distance}")

            if dist < danger_distance or boxes_overlap(f_box, m_box):
                threat_detected = True
                cv2.line(frame, f_center, m_center, (0, 0, 255), 2)
                print("THREAT DETECTED: Close or overlapping")

    if threat_detected:
        cv2.putText(frame, "!!! POSSIBLE THREAT DETECTED !!!", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Threat", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Women Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
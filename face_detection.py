import cv2
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (118, 223, 111), 3)
    
    
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(0):
        break


cap.release()
cv2.destroyAllWindows()
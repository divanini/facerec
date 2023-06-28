from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            emotion = detect_emotion(frame)
            frame = draw_emotion(frame, emotion)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_emotion(frame):
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(detected_faces) > 0:
        face = frame[detected_faces[0][1]:detected_faces[0][1] + detected_faces[0][3], detected_faces[0][0]:detected_faces[0][0] + detected_faces[0][2]]
        emotion = DeepFace.analyze(face, actions=['emotion'])
        return emotion['dominant_emotion']
    else:
        return ""

def draw_emotion(frame, emotion):
    if emotion != "":
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

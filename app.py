from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

class PushupDetector:
    def __init__(self, model_path='pushup_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.pushup_count = 0
        self.position = "up"
        self.prev_prediction = "up"
        self.confidence_threshold = 0.7

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        return processed

    def process_frame(self, frame):
        original_frame = frame.copy()
        
        processed_frame = self.preprocess_frame(frame)
        
        prediction = self.model.predict(np.expand_dims(processed_frame, axis=0))[0]
        position = "up" if prediction[0] > prediction[1] else "down"
        confidence = max(prediction)

        if confidence > self.confidence_threshold:
            if position == "down" and self.prev_prediction == "up":
                self.pushup_count += 1
            self.prev_prediction = position

        cv2.putText(original_frame, f'Count: {self.pushup_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_frame, f'Position: {position}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_frame, f'Confidence: {confidence:.2f}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return original_frame

detector = PushupDetector()
camera = cv2.VideoCapture(0)  # This opens the default webcam (usually the built-in camera)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detector.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
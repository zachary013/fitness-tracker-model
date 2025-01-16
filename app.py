from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

class ExerciseDetector:
    def __init__(self, model_path='exercise_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.exercise_counts = {  # Counter for each exercise
            "jumping_jack": 0,
            "bicep_curl": 0,
            "plank": 0,
            "lunge": 0,
            "pushup": 0,
        }
        self.current_exercise = None
        self.prev_exercise = None
        self.confidence_threshold = 0.7

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        return processed

    def process_frame(self, frame):
        original_frame = frame.copy()

        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(np.expand_dims(processed_frame, axis=0))[0]
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx]

        # Map index to exercise label
        exercises = list(self.exercise_counts.keys())
        detected_exercise = exercises[max_idx]

        if confidence > self.confidence_threshold:
            self.current_exercise = detected_exercise
            if self.current_exercise != self.prev_exercise:
                self.exercise_counts[self.current_exercise] += 1
            self.prev_exercise = self.current_exercise

        # Display information on the frame
        cv2.putText(original_frame, f'Exercise: {self.current_exercise}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_frame, f'Confidence: {confidence:.2f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display counters for all exercises
        y_offset = 110
        for exercise, count in self.exercise_counts.items():
            cv2.putText(original_frame, f'{exercise}: {count}', (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 40

        return original_frame

detector = ExerciseDetector()
camera = cv2.VideoCapture(0)

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

from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

app = Flask(__name__)

class PushupDetector:
    def __init__(self, model_path='pushup_model.h5'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        try:
            self.model = tf.keras.models.load_model(model_path)
        except:
            print(f"Error: Could not load model from {model_path}")
            self.model = None

        self.pushup_count = 0
        self.prev_position = "up"
        self.current_position = "up"
        self.form_quality = "Good"

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Get the model prediction
            if self.model:
                processed = cv2.resize(frame, (224, 224))
                processed = processed / 255.0
                prediction = self.model.predict(np.expand_dims(processed, axis=0))[0]
                self.current_position = "down" if prediction[0] > 0.5 else "up"
                
                # Count push-ups
                if self.current_position == "down" and self.prev_position == "up":
                    self.pushup_count += 1
                self.prev_position = self.current_position
                
                # Check form quality based on pose landmarks
                shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                
                # Simple form check based on body alignment
                if abs(shoulder.y - hip.y) < 0.1 and abs(hip.y - ankle.y) < 0.2:
                    self.form_quality = "Good"
                else:
                    self.form_quality = "Needs Improvement"

        return frame

    def get_stats(self):
        return {
            'count': self.pushup_count,
            'position': self.current_position,
            'form': self.form_quality
        }

detector = PushupDetector()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
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

@app.route('/get_stats')
def get_stats():
    return jsonify(detector.get_stats())

if __name__ == '__main__':
    app.run(debug=True)

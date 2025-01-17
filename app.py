# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
from threading import Lock
import atexit
from mqtt_config import MQTT_CONFIG

app = Flask(__name__)


class ExerciseDetector:
    def __init__(self, model_path='exercise_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.current_exercise = None
        self.confidence_threshold = 0.7
        self._lock = Lock()

        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 1000
        self.is_moving = False
        self.is_active = False
        self.form_quality = "Good"

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id=MQTT_CONFIG['client_id'])
        self.setup_mqtt()

        # Track last state to avoid duplicate messages
        self.last_state = None

    def setup_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Successfully connected to MQTT broker")
            else:
                print(f"Failed to connect to MQTT broker with code: {rc}")
                # Print meaning of return code for debugging
                codes = {
                    1: "incorrect protocol version",
                    2: "invalid client identifier",
                    3: "server unavailable",
                    4: "bad username or password",
                    5: "not authorized"
                }
                print(f"Error means: {codes.get(rc, 'unknown error')}")

        def on_disconnect(client, userdata, rc):
            if rc != 0:
                print(f"Unexpected MQTT disconnection with code: {rc}. Will auto-reconnect")

        def on_publish(client, userdata, mid):
            print(f"Message {mid} published successfully")

        try:
            # Set callbacks
            self.mqtt_client.on_connect = on_connect
            self.mqtt_client.on_disconnect = on_disconnect
            self.mqtt_client.on_publish = on_publish

            # Enable logging
            self.mqtt_client.enable_logger()

            print(f"Connecting to MQTT broker at {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
            self.mqtt_client.connect(
                MQTT_CONFIG['broker'],
                MQTT_CONFIG['port'],
                MQTT_CONFIG['keepalive']
            )
            self.mqtt_client.loop_start()

        except Exception as e:
            print(f"Failed to connect to MQTT broker: {str(e)}")

    def publish_state(self):
        try:
            current_state = {
                'exercise': self.current_exercise,
                'is_moving': bool(self.is_moving),  # Convert NumPy bool_ to Python bool
                'form_quality': self.form_quality,
                'timestamp': int(time.time())
            }

            # Only publish if state has changed
            if current_state != self.last_state:
                # Print for debugging
                print("Publishing state:", current_state)

                topic = MQTT_CONFIG['topics']['exercise_state']

                result = self.mqtt_client.publish(
                    topic,
                    json.dumps(current_state),
                    qos=MQTT_CONFIG['qos']
                )

                if result.rc != 0:
                    print(f"Failed to publish message. Error code: {result.rc}")
                else:
                    print(f"Successfully published to {topic}")
                    self.last_state = current_state

        except Exception as e:
            print(f"Error publishing state: {str(e)}")
            print("Current state:", current_state)  # Print the state for debugging

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_score = np.sum(thresh) / 255
        self.prev_frame = gray
        return bool(motion_score > self.motion_threshold)  # Convert to Python bool

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        return processed

    def process_frame(self, frame):
        original_frame = frame.copy()

        # Update motion state
        self.is_moving = self.detect_motion(frame)

        # Process frame for exercise detection
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)[0]
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx]

        exercises = ["jumping_jack", "bicep_curl", "plank", "lunge", "pushup"]

        if confidence > self.confidence_threshold:
            self.current_exercise = exercises[max_idx]

        # Publish current state to MQTT
        self.publish_state()

        # Draw information on frame
        self._draw_info(original_frame, confidence)
        return original_frame

    def _draw_info(self, frame, confidence):
        motion_status = "Moving" if self.is_moving else "Still"
        cv2.putText(frame, f'Motion: {motion_status}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Exercise: {self.current_exercise}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Form: {self.form_quality}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def cleanup(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


detector = ExerciseDetector()
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if detector.is_active:
                frame = detector.process_frame(frame)
            else:
                cv2.putText(frame, "Press Start to begin", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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


@app.route('/start_detection')
def start_detection():
    detector.is_active = True
    return jsonify({'status': 'started'})


@app.route('/stop_detection')
def stop_detection():
    detector.is_active = False
    return jsonify({'status': 'stopped'})


def cleanup():
    print("Cleaning up resources...")
    camera.release()
    detector.cleanup()
    cv2.destroyAllWindows()


atexit.register(cleanup)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        cleanup()
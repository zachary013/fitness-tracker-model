import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import blynklib
import paho.mqtt.client as mqtt
import json
import time
from threading import Thread, Lock
from flask import Flask, render_template, Response, jsonify
import atexit
from mqtt_config import MQTT_CONFIG, BLYNK_CONFIG

# Flask setup  
app = Flask(__name__)

# Blynk setup
blynk = blynklib.Blynk(BLYNK_CONFIG['token'])

# MQTT setup
mqtt_client = mqtt.Client(client_id=MQTT_CONFIG['client_id'])

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_CONFIG['topics']['pushups'])

def on_mqtt_disconnect(client, userdata, rc):
    print(f"Disconnected from MQTT broker with result code {rc}")
    if rc != 0:
        print("Unexpected disconnection. Attempting to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"MQTT Reconnection failed: {e}")
            time.sleep(5)

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_disconnect = on_mqtt_disconnect

try:
    mqtt_client.connect(MQTT_CONFIG['broker'], MQTT_CONFIG['port'], MQTT_CONFIG['keepalive'])
    mqtt_client.loop_start()
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")

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
        except Exception as e:
            print(f"Error: Could not load model from {model_path}: {e}")
            self.model = None

        self.pushup_count = 0
        self.prev_position = "up"
        self.current_position = "up"
        self.form_quality = "Good"
        self._lock = Lock()

    def increment_count(self):
        with self._lock:
            self.pushup_count += 1

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                if self.model:
                    try:
                        processed = cv2.resize(frame, (224, 224))
                        processed = processed / 255.0
                        prediction = self.model.predict(np.expand_dims(processed, axis=0))[0]
                        self.current_position = "down" if prediction[0] > 0.5 else "up"
                        
                        if self.current_position == "down" and self.prev_position == "up":
                            self.increment_count()
                        self.prev_position = self.current_position
                    except Exception as e:
                        print(f"Error in model prediction: {e}")
                        
                try:
                    shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                    ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    
                    if abs(shoulder.y - hip.y) < 0.1 and abs(hip.y - ankle.y) < 0.2:
                        self.form_quality = "Good"
                    else:
                        self.form_quality = "Needs Improvement"
                except Exception as e:
                    print(f"Error in form quality check: {e}")
                    
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return frame

    def get_stats(self):
        stats = {
            'count': self.pushup_count,
            'position': self.current_position,
            'form': self.form_quality,
            'timestamp': int(time.time())
        }
        
        try:
            mqtt_client.publish(MQTT_CONFIG['topics']['pushups'], json.dumps(stats), qos=MQTT_CONFIG['qos'])
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
        
        try:
            blynk.virtual_write(BLYNK_CONFIG['pins']['pushup_count'], stats['count'])
            blynk.virtual_write(BLYNK_CONFIG['pins']['position'], stats['position'])
            form_quality_value = 100 if stats['form'] == "Good" else 50
            blynk.virtual_write(BLYNK_CONFIG['pins']['form_quality'], form_quality_value)
        except Exception as e:
            print(f"Error updating Blynk: {e}")
        
        return stats

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

def run_blynk():
    while True:
        try:
            blynk.run()
        except Exception as e:
            print(f"Blynk error: {e}")
            time.sleep(5)

def cleanup():
    print("Cleaning up resources...")
    camera.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    cv2.destroyAllWindows()

atexit.register(cleanup)

if __name__ == '__main__':
    try:
        blynk_thread = Thread(target=run_blynk)
        blynk_thread.daemon = True
        blynk_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        cleanup()


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime

class DataCollector:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.up_dir = os.path.join(output_dir, "up")
        self.down_dir = os.path.join(output_dir, "down")
        
        os.makedirs(self.up_dir, exist_ok=True)
        os.makedirs(self.down_dir, exist_ok=True)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.frame_count = {'up': 0, 'down': 0}

    def collect_data(self):
        print("\n=== Push-up Data Collection ===")
        print("Press 'u' to capture UP position")
        print("Press 'd' to capture DOWN position")
        print("Press 'q' to quit")
        print("Aim for at least 30 images of each position")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Draw skeleton
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Display counts
            cv2.putText(frame, f'UP frames: {self.frame_count["up"]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'DOWN frames: {self.frame_count["down"]}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('u'):
                self._save_frame(frame, "up")
            elif key == ord('d'):
                self._save_frame(frame, "down")
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Train model if enough data
        if min(self.frame_count.values()) >= 20:
            print("\nSufficient data collected! Training model...")
            self._train_model()
        else:
            print("\nNot enough data collected. Need at least 20 images per position.")

    def _save_frame(self, frame, position):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = self.up_dir if position == "up" else self.down_dir
        filename = os.path.join(output_dir, f"{position}_{timestamp}.jpg")
        
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(filename, frame)
        self.frame_count[position] += 1
        print(f"Saved {position} position frame ({self.frame_count[position]})")

    def _train_model(self):
        # Data preprocessing
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        train_generator = datagen.flow_from_directory(
            self.output_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        # Build model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile and train
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("\nTraining model... This may take a few minutes.")
        model.fit(train_generator, epochs=10, verbose=1)
        
        # Save model
        model.save('pushup_model.h5')
        print("\nModel saved as 'pushup_model.h5'")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()

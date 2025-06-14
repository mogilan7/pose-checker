from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
pose = mp.solutions.pose.Pose(static_image_mode=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return jsonify({'error': 'No person detected'}), 200

    landmarks = {
        i: {
            'x': round(lm.x, 4),
            'y': round(lm.y, 4),
            'z': round(lm.z, 4)
        }
        for i, lm in enumerate(results.pose_landmarks.landmark)
    }
    return jsonify({'landmarks': landmarks})
import cv2
import numpy as np
from src.classifier import classify_waste, WASTE_CATEGORIES

def draw_detection(frame, waste_type, confidence, info):
    """Draw detection results on the frame."""
    h, w = frame.shape[:2]
    color = info["color"]

    # Draw border around frame
    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 3)

    # Draw top banner
    cv2.rectangle(frame, (0, 0), (w, 70), color, -1)

    # Waste type text
    cv2.putText(
        frame,
        f"{info['emoji']} {waste_type}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )

    # Confidence text
    cv2.putText(
        frame,
        f"Confidence: {confidence}%",
        (w - 220, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Draw bottom banner
    cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)

    # Bin instruction
    cv2.putText(
        frame,
        f"Bin: {info['bin']}",
        (10, h-50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Tip text
    cv2.putText(
        frame,
        f"Tip: {info['tip'][:50]}...",
        (10, h-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    return frame

def get_camera_frame(cap):
    """Read a single frame from camera."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def process_frame(frame):
    """Process a frame and return detection results."""
    waste_type, confidence, info = classify_waste(frame)
    annotated_frame = draw_detection(frame.copy(), waste_type, confidence, info)
    return annotated_frame, waste_type, confidence, info
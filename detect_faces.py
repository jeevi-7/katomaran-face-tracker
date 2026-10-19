
from pathlib import Path
import cv2
from ultralytics import YOLO

# Project folder
BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "sample.mp4"
MODEL_PATH = BASE_DIR / "models" / "yolov8n-face.pt"
OUTPUT_PATH = BASE_DIR / "face_detection_output.mp4"


def main():
    # Check input files
    if not VIDEO_PATH.exists():
        print("ERROR: sample.mp4 not found in the project folder.")
        return

    if not MODEL_PATH.exists():
        print("ERROR: Face model not found.")
        print("Expected location:", MODEL_PATH)
        return

    # Load YOLO face model
    model = YOLO(str(MODEL_PATH))

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        print("ERROR: Could not create output video.")
        return

    frame_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Detect faces
            results = model.predict(
                frame,
                conf=0.40,
                verbose=False
            )

            # Draw bounding boxes
            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

            frame_count += 1

            cv2.imshow(
                "Katomaran Face Detection",
                annotated_frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Processed frames:", frame_count)
    print("Output saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

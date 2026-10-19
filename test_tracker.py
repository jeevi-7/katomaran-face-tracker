
import numpy as np
from tracker import FaceTracker


def main():
    tracker = FaceTracker(max_missing_frames=2)

    # Create a fake 512-dimensional face embedding
    embedding = np.ones(512, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)

    # Fake cropped face image: 90 x 90 pixels
    crop = np.zeros((90, 90, 3), dtype=np.uint8)

    detection = {
        "bbox": (10, 10, 100, 100),
        "embedding": embedding,
        "visitor_id": "visitor_001",
        "crop": crop,
    }

    print("=== Face Tracker Test ===")

    # Frame 1: detect the face
    matches, departed = tracker.update([detection])
    print("Frame 1 — detected:", matches)

    # Frame 2: detect the same face again
    matches, departed = tracker.update([detection])
    print("Frame 2 — detected:", matches)

    # First missing frame
    _, departed = tracker.update([])
    print("Missing frame 1 — departed:", departed)

    # Second missing frame
    _, departed = tracker.update([])
    print("Missing frame 2 — departed:", departed)

    # Third missing frame: track should time out
    _, departed = tracker.update([])

    departed_summary = [
        {
            "track_id": track["track_id"],
            "visitor_id": track["visitor_id"],
        }
        for track in departed
    ]

    print("Missing frame 3 — departed:", departed_summary)

    assert departed_summary == [
        {
            "track_id": 1,
            "visitor_id": "visitor_001",
        }
    ]

    print("\nSUCCESS: Tracker test passed!")


if __name__ == "__main__":
    main()


import cv2
import json
import logging
import threading
import time
import numpy as np

from pathlib import Path
from datetime import datetime
from queue import Queue, Empty, Full

from insightface.app import FaceAnalysis

from database import (
    load_visitors,
    save_visitor,
    save_event,
    get_unique_visitor_count,
)

from tracker import FaceTracker


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

VIDEO_PATH = BASE_DIR / CONFIG.get("video_source", "sample.mp4")
OUTPUT_PATH = BASE_DIR / "recognition_tracking_output.mp4"

SKIP_FRAMES = max(
    1,
    int(CONFIG.get("detection_skip_frames", 8))
)

SIMILARITY_THRESHOLD = float(
    CONFIG.get("recognition_threshold", 0.45)
)

MAX_MISSING_FRAMES = max(
    1,
    int(CONFIG.get("tracker_max_missing_frames", 15))
)

IMAGE_ROOT = BASE_DIR / CONFIG.get(
    "image_output_dir", "output_images"
)

LOG_DIR = BASE_DIR / CONFIG.get("log_dir", "logs")

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    filename=str(LOG_DIR / "events.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger("visitor_tracker")


# ==================================================
# 3. LOAD RECOGNITION MODEL
# ==================================================

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
)

print("InsightFace model loaded.")


# ==================================================
# 4. LOAD KNOWN VISITORS
# ==================================================

known_visitors = load_visitors()

print(f"Loaded {len(known_visitors)} registered visitors.")


# ==================================================
# 5. VISITOR IDENTIFICATION
# ==================================================

def identify_visitor(embedding):
    """
    Match a face embedding against registered visitors.
    Register a new visitor if no sufficiently similar
    existing visitor is found.
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    embedding_norm = np.linalg.norm(embedding)

    if embedding_norm == 0:
        return None, 0.0

    normalized_embedding = embedding / embedding_norm

    best_visitor_id = None
    best_similarity = -1.0

    for visitor_id, saved_embedding in known_visitors.items():
        saved_embedding = np.asarray(
            saved_embedding,
            dtype=np.float32,
        )

        saved_norm = np.linalg.norm(saved_embedding)

        if saved_norm == 0:
            continue

        normalized_saved = saved_embedding / saved_norm

        similarity = float(
            np.dot(normalized_embedding, normalized_saved)
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_visitor_id = visitor_id

    if (
        best_visitor_id is not None
        and best_similarity >= SIMILARITY_THRESHOLD
    ):
        return best_visitor_id, best_similarity

    visitor_id = f"visitor_{len(known_visitors) + 1:03d}"

    # Avoid reusing an ID if visitor IDs are not consecutive.
    while visitor_id in known_visitors:
        visitor_id = f"visitor_{len(known_visitors) + 2:03d}"
        if visitor_id in known_visitors:
            # Use the next available ID.
            number = 1
            while f"visitor_{number:03d}" in known_visitors:
                number += 1
            visitor_id = f"visitor_{number:03d}"
            break

    save_visitor(visitor_id, embedding)

    known_visitors[visitor_id] = embedding.copy()

    logger.info("Registered new visitor: %s", visitor_id)

    print(f"New visitor registered: {visitor_id}")

    return visitor_id, 1.0


# ==================================================
# 6. SAVE ENTRY OR EXIT EVENT
# ==================================================

def record_event(visitor_id, event_type, crop):
    """
    Save a visitor crop and create one database event.
    """

    if visitor_id is None or crop is None or crop.size == 0:
        logger.warning(
            "Skipped invalid event: visitor=%s type=%s",
            visitor_id,
            event_type,
        )
        return False

    if event_type not in ("entry", "exit"):
        raise ValueError("Invalid event type")

    now = datetime.now()

    day_folder = IMAGE_ROOT / now.strftime("%Y-%m-%d")
    day_folder.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{visitor_id}_{event_type}_"
        f"{now.strftime('%H%M%S_%f')}.jpg"
    )

    image_path = day_folder / filename

    if not cv2.imwrite(str(image_path), crop):
        logger.error("Failed to save image: %s", image_path)
        return False

    try:
        # Store a path relative to the project directory.
        relative_path = str(image_path.relative_to(BASE_DIR))

        save_event(
            visitor_id,
            event_type,
            relative_path,
        )

        logger.info(
            "%s event: %s | image=%s",
            event_type.upper(),
            visitor_id,
            relative_path,
        )

        print(
            f"{event_type.upper()}: {visitor_id} "
            f"| {relative_path}"
        )

        return True

    except Exception:
        logger.exception(
            "Could not save %s event for %s",
            event_type,
            visitor_id,
        )
        raise


# ==================================================
# 7. BACKGROUND FACE DETECTION
# ==================================================

detection_queue = Queue(maxsize=1)

result_lock = threading.Lock()

result_state = {
    "frame_number": -1,
    "detections": [],
    "error": None,
}

stop_event = threading.Event()


def detection_worker():
    """
    Perform model inference in a background thread.

    Database writes and visitor identification remain
    in the main thread.
    """

    while not stop_event.is_set() or not detection_queue.empty():

        try:
            frame_number, frame = detection_queue.get(
                timeout=0.1
            )
        except Empty:
            continue

        try:
            faces = app.get(frame)

            detections = []

            height, width = frame.shape[:2]

            for face in faces:
                bbox = np.asarray(
                    face.bbox,
                    dtype=np.float32,
                )

                x1, y1, x2, y2 = bbox.astype(int)

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()

                if crop.size == 0:
                    continue

                detections.append({
                    "bbox": bbox,
                    "embedding": np.asarray(
                        face.embedding,
                        dtype=np.float32,
                    ).copy(),
                    "crop": crop,
                    "score": float(
                        getattr(face, "det_score", 0.0)
                    ),
                })

            with result_lock:
                result_state["frame_number"] = frame_number
                result_state["detections"] = detections
                result_state["error"] = None

        except Exception as error:
            logger.exception("Face detection failed")

            with result_lock:
                result_state["error"] = str(error)

        finally:
            detection_queue.task_done()


# ==================================================
# 8. VIDEO INITIALIZATION
# ==================================================

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_PATH}"
    )


# ==================================================
# 9. TRACKER AND EVENT STATE
# ==================================================

tracker = FaceTracker(
    max_missing_frames=MAX_MISSING_FRAMES,
    similarity_threshold=SIMILARITY_THRESHOLD,
)

# Visitors whose entry has been recorded and whose exit
# has not yet been recorded during this run.
active_logged_visitors = set()

# Most recent usable crop for each visitor.
last_crops = {}

last_detection_frame = -1
frame_number = 0

# Cache the count instead of querying MySQL every frame.
try:
    visitor_count = get_unique_visitor_count()
except Exception:
    visitor_count = len(known_visitors)

worker = threading.Thread(
    target=detection_worker,
    daemon=True,
)

worker.start()


# ==================================================
# 10. MAIN VIDEO LOOP
# ==================================================

print("Starting face recognition and tracking...")
print("Press Q in the video window to stop.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Submit frames without allowing a backlog to grow.
        if (frame_number - 1) % SKIP_FRAMES == 0:

            if detection_queue.full():
                try:
                    detection_queue.get_nowait()
                    detection_queue.task_done()
                except Empty:
                    pass

            try:
                detection_queue.put_nowait(
                    (frame_number, frame.copy())
                )
            except Full:
                pass

        # Get the newest completed detection result.
        with result_lock:
            result_frame = result_state["frame_number"]
            result_detections = result_state["detections"]
            worker_error = result_state["error"]

        if worker_error:
            cv2.putText(
                frame,
                "Detection warning - check logs",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        # Update tracking only when a new inference result
        # has actually arrived.
        if result_frame > last_detection_frame:

            last_detection_frame = result_frame

            prepared_detections = []

            for detection in result_detections:
                visitor_id, similarity = identify_visitor(
                    detection["embedding"]
                )

                prepared_detections.append({
                    "bbox": detection["bbox"],
                    "embedding": detection["embedding"],
                    "visitor_id": visitor_id,
                    "crop": detection["crop"],
                    "score": detection["score"],
                })

            active_matches, departed_tracks = tracker.update(
                prepared_detections
            )

            # Update visitor crops from currently tracked faces.
            for track_id, track in tracker.get_active_tracks():
                visitor_id = track.get("visitor_id")
                crop = track.get("crop")

                if visitor_id and crop is not None:
                    last_crops[visitor_id] = crop.copy()

            # Record entries once per active visitor.
            for track_id, detection_index in active_matches:

                detection = prepared_detections[detection_index]
                visitor_id = detection["visitor_id"]
                crop = detection["crop"]

                if visitor_id is None:
                    continue

                last_crops[visitor_id] = crop.copy()

                if visitor_id not in active_logged_visitors:
                    saved = record_event(
                        visitor_id,
                        "entry",
                        crop,
                    )

                    if saved:
                        active_logged_visitors.add(visitor_id)

            # Process tracks that have timed out.
            # If another active track has the same visitor ID,
            # don't record an exit yet.
            active_track_visitor_ids = {
                track.get("visitor_id")
                for _, track in tracker.get_active_tracks()
                if track.get("visitor_id") is not None
            }

            for departed_item in departed_tracks:

                if (
                    isinstance(departed_item, tuple)
                    and len(departed_item) == 2
                ):
                    _, departed_track = departed_item
                else:
                    departed_track = departed_item

                visitor_id = departed_track.get("visitor_id")

                if visitor_id is None:
                    continue

                if visitor_id in active_track_visitor_ids:
                    continue

                if visitor_id in active_logged_visitors:
                    crop = (
                        departed_track.get("crop")
                        or last_crops.get(visitor_id)
                    )

                    if crop is not None:
                        saved = record_event(
                            visitor_id,
                            "exit",
                            crop,
                        )

                        if saved:
                            active_logged_visitors.discard(
                                visitor_id
                            )
                            last_crops.pop(visitor_id, None)

            # Update the cached count only when recognition
            # has processed new detections.
            visitor_count = len(known_visitors)

        # --------------------------------------------------
        # DRAW TRACKS ON EVERY VIDEO FRAME
        # --------------------------------------------------

        for track_id, track in tracker.get_active_tracks():

            bbox = np.asarray(
                track["bbox"],
                dtype=np.int32,
            )

            x1, y1, x2, y2 = bbox

            visitor_id = track.get(
                "visitor_id",
                "Unknown",
            )

            score = track.get("score", 0.0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{visitor_id} | Track {track_id} "
                f"| {score:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Registered Visitors: {visitor_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        writer.write(frame)

        cv2.imshow(
            "Katomaran Face Recognition",
            frame,
        )

        if cv2.waitKey(
            max(1, int(1000 / fps))
        ) & 0xFF == ord("q"):
            break


# ==================================================
# 11. CLEANUP AND FINAL EXIT EVENTS
# ==================================================

finally:
    stop_event.set()

    worker.join(timeout=10)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Close any visitor sessions still active when the video
    # ends or the user presses Q.
    for visitor_id in list(active_logged_visitors):

        crop = last_crops.get(visitor_id)

        if crop is not None:
            try:
                if record_event(
                    visitor_id,
                    "exit",
                    crop,
                ):
                    active_logged_visitors.discard(visitor_id)

            except Exception:
                logger.exception(
                    "Failed final exit event for %s",
                    visitor_id,
                )

    print("\nProcessing finished.")
    print(f"Video output: {OUTPUT_PATH}")
    print(f"Registered visitors: {visitor_count}")

    logger.info("Video processing finished.")
    logger.info("Output video: %s", OUTPUT_PATH)

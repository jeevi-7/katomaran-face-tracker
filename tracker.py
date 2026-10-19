
import logging
import numpy as np


class FaceTracker:
    """Track detected faces between frames and report departed tracks."""

    def __init__(
        self,
        max_missing_frames=15,
        similarity_threshold=0.45,
        iou_threshold=0.2,
    ):
        self.max_missing_frames = max_missing_frames
        self.similarity_threshold = similarity_threshold
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_track_id = 1
        self.logger = logging.getLogger("face_tracker")

    @staticmethod
    def cosine_similarity(a, b):
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return -1.0
        return float(np.dot(a, b) / denominator)

    @staticmethod
    def calculate_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        """Update tracks using a list of face detections."""

        for track in self.tracks.values():
            track["missing_frames"] += 1

        unmatched = set(range(len(detections)))
        matched_tracks = set()
        active_matches = []
        candidates = []

        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                iou = self.calculate_iou(
                    track["bbox"], detection["bbox"]
                )
                similarity = self.cosine_similarity(
                    track["embedding"], detection["embedding"]
                )

                same_visitor = (
                    track["visitor_id"] == detection["visitor_id"]
                )

                if iou >= self.iou_threshold or (
                    same_visitor
                    and similarity >= self.similarity_threshold
                ):
                    score = iou + max(0.0, similarity) * 0.5
                    candidates.append((score, track_id, index))

        candidates.sort(reverse=True)

        for _, track_id, index in candidates:
            if track_id in matched_tracks or index not in unmatched:
                continue

            detection = detections[index]
            track = self.tracks[track_id]

            track["bbox"] = tuple(detection["bbox"])
            track["embedding"] = detection["embedding"].copy()
            track["visitor_id"] = detection["visitor_id"]
            track["crop"] = detection["crop"].copy()
            track["score"] = detection.get("score", 0.0)
            track["missing_frames"] = 0

            active_matches.append((track_id, index))
            matched_tracks.add(track_id)
            unmatched.remove(index)

            self.logger.info(
                "TRACKING | track_id=%s | visitor_id=%s",
                track_id,
                track["visitor_id"],
            )

        for index in sorted(unmatched):
            detection = detections[index]
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                "bbox": tuple(detection["bbox"]),
                "embedding": detection["embedding"].copy(),
                "visitor_id": detection["visitor_id"],
                "crop": detection["crop"].copy(),
                "score": detection.get("score", 0.0),
                "missing_frames": 0,
            }

            active_matches.append((track_id, index))

            self.logger.info(
                "TRACK_CREATED | track_id=%s | visitor_id=%s",
                track_id,
                detection["visitor_id"],
            )

        departed_tracks = []

        for track_id in list(self.tracks):
            track = self.tracks[track_id]

            if track["missing_frames"] > self.max_missing_frames:
                departed_tracks.append({
                    "track_id": track_id,
                    "visitor_id": track["visitor_id"],
                    "bbox": track["bbox"],
                    "embedding": track["embedding"].copy(),
                    "crop": track["crop"].copy(),
                    "score": track["score"],
                })

                self.logger.info(
                    "TRACK_DEPARTED | track_id=%s | visitor_id=%s",
                    track_id,
                    track["visitor_id"],
                )

                del self.tracks[track_id]

        return active_matches, departed_tracks

    def get_active_tracks(self):
        """Return recent tracks for drawing boxes between detections."""
        return [
            (track_id, track)
            for track_id, track in self.tracks.items()
            if track["missing_frames"] <= self.max_missing_frames
        ]

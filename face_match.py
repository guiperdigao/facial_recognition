"""
Face matching script:
1. Plays tono.mp3, then opens the webcam
2. Auto-captures when the detected face has been still for 3 s
   (shows "Permanece quieto" while the face is moving)
3. Compares the captured face against all .png portraits in ./portraits/
4. Displays the best match side-by-side for 15 s with a countdown
5. Plays tono.mp3 again and loops back to the camera (ESC to quit)
"""

import sys
import subprocess
import time
import cv2
import face_recognition
import numpy as np
from pathlib import Path

PORTRAITS_DIR        = Path(__file__).parent / "portraits"
TONO_PATH            = Path(__file__).parent / "tono.mp3"
WINDOW_CAPTURE       = "Camera — ESC para salir"
WINDOW_RESULT        = "Mejor coincidencia"

STILLNESS_THRESHOLD  = 20   # max face-center movement in 1/4-scale pixels
STILLNESS_DURATION_S = 3.0  # seconds of stillness before auto-capture
RESULT_DISPLAY_S     = 15   # seconds to show the result


# ── Audio ─────────────────────────────────────────────────────────────────────

def play_sound(path: Path) -> None:
    """Play an MP3 non-blocking via afplay (macOS built-in)."""
    if path.exists():
        subprocess.Popen(["afplay", "-t", "1.2", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Portrait loading ──────────────────────────────────────────────────────────

def load_portraits(portraits_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load every .png from portraits_dir and return (path, image_bgr, encoding)."""
    portraits = []
    for png in sorted(portraits_dir.glob("*.png")):
        img_bgr = cv2.imread(str(png))
        if img_bgr is None:
            print(f"[warn] Could not read {png}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if not encs:
            print(f"[warn] No face found in {png.name}, skipping.")
            continue
        portraits.append((str(png), img_bgr, encs[0]))
        print(f"[info] Loaded portrait: {png.name}")
    return portraits


# ── Matching ──────────────────────────────────────────────────────────────────

def find_best_match(
    captured_enc: np.ndarray,
    portraits: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[str, np.ndarray, float] | None:
    """Return (path, image_bgr, distance) for the closest portrait, or None."""
    if not portraits:
        return None
    paths, images, encs = zip(*portraits)
    distances = face_recognition.face_distance(list(encs), captured_enc)
    best_idx = int(np.argmin(distances))
    return paths[best_idx], images[best_idx], float(distances[best_idx])


# ── Drawing helpers ───────────────────────────────────────────────────────────

def annotate_bottom(image: np.ndarray, label: str, color=(0, 200, 0)) -> np.ndarray:
    """Draw a label banner at the bottom of a copy of image."""
    img = image.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(img, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def annotate_top(image: np.ndarray, label: str, color=(200, 200, 200)) -> np.ndarray:
    """Draw a label banner at the top of a copy of image."""
    img = image.copy()
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return img


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize both images to the same height and combine side-by-side."""
    target_h = 480

    def resize_to_height(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    left_r  = resize_to_height(left, target_h)
    right_r = resize_to_height(right, target_h)
    sep = np.zeros((target_h, 4, 3), dtype=np.uint8)
    return np.hstack([left_r, sep, right_r])


# ── Camera loop with stillness detection ──────────────────────────────────────

def capture_with_stillness(cap: cv2.VideoCapture) -> np.ndarray | None:
    """
    Show live camera feed and return a frame once the face has been still
    for STILLNESS_DURATION_S seconds.  Returns None if ESC is pressed.
    """
    still_since: float | None = None
    last_center: tuple[int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[warn] Failed to grab frame, retrying…")
            continue

        # Run face detection on a 1/4-scale image for speed
        small    = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)

        now      = time.time()
        captured = False

        if locations:
            top, right, bottom, left = locations[0]
            cx, cy = (left + right) // 2, (top + bottom) // 2

            # Detect movement relative to the previous frame
            if last_center is not None:
                moved = abs(cx - last_center[0]) + abs(cy - last_center[1]) > STILLNESS_THRESHOLD
                if moved:
                    still_since = None
                elif still_since is None:
                    still_since = now
            else:
                still_since = now   # first frame a face appears

            last_center = (cx, cy)

            # Draw face box (scale coordinates back to full resolution)
            cv2.rectangle(frame,
                          (left * 4, top * 4),
                          (right * 4, bottom * 4),
                          (0, 255, 0), 2)

            if still_since is not None:
                elapsed   = now - still_since
                captured  = elapsed >= STILLNESS_DURATION_S
                remaining = max(0.0, STILLNESS_DURATION_S - elapsed)
                if captured:
                    label, color = "Capturando!", (0, 255, 0)
                else:
                    label, color = f"Quieto... {remaining:.1f}s", (0, 255, 0)
            else:
                label, color = "Permanece quieto", (0, 0, 220)
        else:
            last_center = None
            still_since = None
            label, color = "Permanece quieto", (0, 0, 220)

        display = annotate_bottom(frame, label, color=color)
        cv2.imshow(WINDOW_CAPTURE, display)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            return None

        if captured:
            return frame.copy()


# ── Timed result display ──────────────────────────────────────────────────────

def show_result_timed(combined: np.ndarray, duration_s: int = RESULT_DISPLAY_S) -> bool:
    """
    Display the combined result image with a top-bar countdown.
    Returns True to keep looping, False if ESC was pressed.
    """
    end_time = time.time() + duration_s
    while True:
        remaining = int(end_time - time.time())
        if remaining < 0:
            return True

        frame = annotate_top(combined,
                              f"Volviendo en {remaining}s — ESC para salir")
        cv2.imshow(WINDOW_RESULT, frame)

        if cv2.waitKey(200) & 0xFF == 27:   # ESC
            return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PORTRAITS_DIR.exists():
        sys.exit(f"[error] Portraits folder not found: {PORTRAITS_DIR}")

    print(f"[info] Loading portraits from {PORTRAITS_DIR} …")
    portraits = load_portraits(PORTRAITS_DIR)
    if not portraits:
        sys.exit("[error] No usable portraits found (need at least one .png with a visible face).")
    print(f"[info] {len(portraits)} portrait(s) ready.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[error] Could not open camera.")

    try:
        while True:
            # ── 1. Play intro tone ────────────────────────────────────────────
            play_sound(TONO_PATH)

            # ── 2. Wait for a still face (auto-capture) ───────────────────────
            print("[info] Waiting for a still face…")
            captured_frame = capture_with_stillness(cap)
            cv2.destroyWindow(WINDOW_CAPTURE)

            if captured_frame is None:
                print("[info] ESC pressed. Quitting.")
                break

            # ── 3. Encode the captured face ───────────────────────────────────
            rgb_captured  = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            captured_encs = face_recognition.face_encodings(rgb_captured)

            if not captured_encs:
                print("[warn] No face detected in captured photo, restarting…")
                continue

            # ── 4. Find best match ────────────────────────────────────────────
            print("[info] Comparing against portraits…")
            result = find_best_match(captured_encs[0], portraits)

            if result is None:
                print("[warn] Matching failed unexpectedly, restarting…")
                continue

            match_path, match_img, distance = result
            similarity_pct = max(0.0, (1.0 - distance)) * 100
            match_name     = Path(match_path).stem
            is_match       = distance < 0.6
            verdict        = "MATCH" if is_match else "NO MATCH"
            verdict_color  = (0, 220, 0) if is_match else (0, 0, 220)

            print(f"\n  Best match : {match_name}")
            print(f"  Distance   : {distance:.4f}  (lower = more similar)")
            print(f"  Similarity : {similarity_pct:.1f}%  →  {verdict}\n")

            left_ann  = annotate_bottom(captured_frame, "Captured",
                                        color=(200, 200, 200))
            right_ann = annotate_bottom(match_img,
                                        f"{match_name}  |  {verdict}  ({similarity_pct:.1f}%)",
                                        color=verdict_color)
            combined  = make_side_by_side(left_ann, right_ann)

            # ── 5. Show result for 15 s ───────────────────────────────────────
            keep_going = show_result_timed(combined, RESULT_DISPLAY_S)
            cv2.destroyWindow(WINDOW_RESULT)

            if not keep_going:
                print("[info] ESC pressed. Quitting.")
                break

            # tone plays at the top of the next loop iteration

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[info] Done.")


if __name__ == "__main__":
    main()

"""Gesture Desk — one-file webcam gesture launcher for Windows.

SETUP (VS Code terminal, using your installed Python 3.13):
    python -m pip install --upgrade mediapipe opencv-python pillow
    python gesture_desk.py

On first run the script downloads Google's hand-landmarker model to
%LOCALAPPDATA%\\GestureDesk. Internet is needed only for that first download.
Edit ACTIONS below to change what each gesture launches. Hold a gesture for one
second to confirm; press Q or Escape to quit, or C to request camera auto-focus.
"""

import math
import os
import platform
import subprocess
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import messagebox

import cv2
import mediapipe as mp
import ctypes

ACTIONS = {
    "Pinky finger": "https://github.com",
    "Thumbs up": "https://www.youtube.com",
    "Fist": "notepad.exe",
    "Middle finger": "calc.exe",
}

CAMERA_INDEX = 0
HOLD_SECONDS = 1.0
LAUNCH_COOLDOWN = 3.0
MIRROR_VIEW = True
WINDOW_TITLE = "Gesture Desk — webcam control"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Finger order: thumb, index, middle, ring, pinky. 1 means extended.
POSES = {
    (1, 1, 1, 1, 1): "Open palm",
    (0, 0, 0, 0, 0): "Fist",
    (0, 1, 0, 0, 0): "Index finger",
    (0, 1, 1, 0, 0): "Peace sign",
    (0, 1, 1, 1, 0): "Three fingers",
    (0, 1, 1, 1, 1): "Four fingers",
    (1, 0, 0, 0, 0): "Thumbs up",
    (0, 0, 0, 0, 1): "Pinky finger",
    (0, 1, 0, 0, 1): "Rock sign",
    (1, 0, 0, 0, 1): "Call me",
    (1, 1, 0, 0, 0): "Index + thumb",
    (0, 0, 1, 0, 0): "Middle finger",
}
# MediaPipe hand landmark index pairs, used to draw a skeleton without the
# removed mp.solutions drawing helper.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def ensure_model():
    """Get the official task model once; keep the source program single-file."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "GestureDesk"
    model = base / "hand_landmarker.task"
    if model.is_file() and model.stat().st_size > 1_000_000:
        return model
    base.mkdir(parents=True, exist_ok=True)
    partial = base / "hand_landmarker.download"
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "GestureDesk/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model file is unexpectedly small")
        partial.replace(model)
        return model
    except Exception as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "Could not download the hand-tracking model. Check your internet connection "
            f"and try again. Model URL: {MODEL_URL}\nDetails: {exc}"
        ) from exc


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_states(landmarks):
    wrist = landmarks[0]
    palm = max(distance(wrist, landmarks[9]), 1e-5)
    states = []
    for tip_i, pip_i in ((8, 6), (12, 10), (16, 14), (20, 18)):
        states.append(int(distance(wrist, landmarks[tip_i]) > distance(wrist, landmarks[pip_i]) + 0.10 * palm))
    thumb_tip = distance(landmarks[4], landmarks[5])
    thumb_joint = distance(landmarks[3], landmarks[5])
    states.insert(0, int(thumb_tip > thumb_joint + 0.16 * palm))
    pinch = distance(landmarks[4], landmarks[8]) < 0.28 * palm
    return tuple(states), pinch


def detect_gesture(landmarks):
    states, pinch = finger_states(landmarks)
    if pinch:
        return "OK pinch"
    return POSES.get(states)


def open_action(action):
    if not action:
        return "Gesture disabled. Edit ACTIONS in the script."
    try:
        if action.startswith(("https://", "http://")):
            import webbrowser
            webbrowser.open(action, new=2)
        elif platform.system() == "Windows":
            os.startfile(action)
        else:
            subprocess.Popen([action], shell=False)
        return f"Opened {action}"
    except Exception as exc:
        return f"Could not open {action}: {exc}"


class GestureDesk:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="#10151c")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("q", lambda _event: self.close())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("c", lambda _event: self.calibrate())

        tk.Label(root, text="GESTURE DESK", font=("Segoe UI", 20, "bold"),
                 fg="#f2f6ff", bg="#10151c").pack(pady=(12, 4))
        self.canvas = tk.Label(root, bg="#080b10")
        self.canvas.pack(padx=14, pady=8)
        self.status = tk.Label(root, text="Starting camera…", font=("Segoe UI", 11),
                               fg="#c7d3e3", bg="#10151c")
        self.status.pack(pady=(0, 4))
        tk.Label(root, text="Move cursor: index finger • click: pinch/release • hold pinky, thumbs-up, fist, or middle finger to launch • M: cursor toggle",
                 font=("Segoe UI", 9), fg="#8292a8", bg="#10151c").pack(pady=(0, 10))
        self.root.bind("m", lambda _event: self.toggle_mode())

        self.running = True
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.running = False
            messagebox.showerror("Camera unavailable", "Could not open webcam. Check Windows camera permissions or change CAMERA_INDEX in the script.")
            root.destroy()
            return

        model_path = ensure_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.current = None
        self.current_since = time.monotonic()
        self.last_launch = 0.0
        self.last_message = "Show one hand to the camera"
        self.last_timestamp_ms = 0
        self.cursor_mode = True
        self.cursor_x = None
        self.cursor_y = None
        self.mouse_down = False
        self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        self.root.after(10, self.update_frame)

    def toggle_mode(self):
        self.cursor_mode = not self.cursor_mode
        self.current = None
        self.current_since = time.monotonic()
        self.status.config(text=("Cursor movement ON — four gesture shortcuts remain active"
                                 if self.cursor_mode else "Cursor movement OFF — four gesture shortcuts remain active"))

    def set_cursor(self, landmarks):
        """Map the mirrored webcam index fingertip to the Windows screen."""
        tip = landmarks[8]
        x = max(0.0, min(1.0, (tip.x - 0.04) / 0.92))
        y = max(0.0, min(1.0, (tip.y - 0.04) / 0.92))
        target_x = int(x * (self.screen_w - 1))
        target_y = int(y * (self.screen_h - 1))
        if self.cursor_x is None:
            self.cursor_x, self.cursor_y = target_x, target_y
        else:
            # Smooth small landmark jitters while keeping pointer movement responsive.
            self.cursor_x = int(0.35 * target_x + 0.65 * self.cursor_x)
            self.cursor_y = int(0.35 * target_y + 0.65 * self.cursor_y)
        ctypes.windll.user32.SetCursorPos(self.cursor_x, self.cursor_y)

    def set_mouse_button(self, down):
        if down == self.mouse_down:
            return
        flag = 0x0002 if down else 0x0004  # left button down/up
        ctypes.windll.user32.mouse_event(flag, 0, 0, 0, 0)
        self.mouse_down = down

    def calibrate(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.last_message = "Requested automatic focus"

    def update_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(text="Camera frame unavailable")
            self.root.after(100, self.update_frame)
            return
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        gesture = None
        landmarks = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            h, w = frame.shape[:2]
            points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (80, 230, 130), 2, cv2.LINE_AA)
            for x, y in points:
                cv2.circle(frame, (x, y), 4, (255, 190, 60), -1, cv2.LINE_AA)
            gesture = detect_gesture(landmarks)

        if self.cursor_mode and landmarks:
            self.set_cursor(landmarks)
            # A thumb/index pinch presses; opening the pinch releases (click).
            self.set_mouse_button(gesture == "OK pinch")
        elif self.mouse_down:
            self.set_mouse_button(False)

        now = time.monotonic()
        if gesture != self.current:
            self.current = gesture
            self.current_since = now
        if gesture:
            held = now - self.current_since
            ready = held >= HOLD_SECONDS
            color = (60, 220, 140) if ready else (0, 190, 255)
            cv2.putText(frame, gesture, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            progress = min(held / HOLD_SECONDS, 1.0)
            cv2.rectangle(frame, (18, 52), (218, 62), (55, 65, 78), -1)
            cv2.rectangle(frame, (18, 52), (18 + int(200 * progress), 62), color, -1)
            if ready and ACTIONS.get(gesture) and now - self.last_launch >= LAUNCH_COOLDOWN:
                self.last_message = open_action(ACTIONS.get(gesture))
                self.last_launch = now
                self.current_since = now + 999
        else:
            cv2.putText(frame, "No recognized gesture", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (170, 180, 195), 2, cv2.LINE_AA)

        from PIL import Image, ImageTk
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((900, 560))
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.configure(image=photo, width=image.width, height=image.height)
        self.canvas.image = photo
        gesture_text = gesture or "No shortcut gesture"
        action = ACTIONS.get(gesture, "") if gesture else ""
        cursor_text = "Cursor ON" if self.cursor_mode else "Cursor OFF"
        self.status.config(text=f"{cursor_text} • {gesture_text} {('→ ' + action) if action else ''} • {self.last_message}")
        self.root.after(15, self.update_frame)

    def close(self):
        if not self.running:
            return
        self.running = False
        if getattr(self, "mouse_down", False):
            self.set_mouse_button(False)
        try:
            self.detector.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        GestureDesk(root)
        if root.winfo_exists():
            root.mainloop()
    except Exception as exc:
        try:
            messagebox.showerror("Gesture Desk error", str(exc))
        except tk.TclError:
            print(f"Gesture Desk error: {exc}")
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()

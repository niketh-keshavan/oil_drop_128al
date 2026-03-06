"""
Millikan Oil Drop Experiment — Video Frame Marker
===================================================
Play back oil drop videos and interactively mark the frames where the
droplet is at the TOP or BOTTOM of each oscillation (reticle crossing).

Output
------
* A JSON file  (<video_stem>_marks.json)  with frame numbers, timestamps,
  and per-droplet voltage.
* Saved frame images in  data/frames/<video_stem>/  for every marked position.

Controls
--------
  Space        Play / Pause
  D  or  →     Step forward 1 frame
  A  or  ←     Step backward 1 frame
  W  or  ↑     Step forward 10 frames
  S  or  ↓     Step backward 10 frames (when paused; otherwise S = save)
  T            Mark current frame as TOP
  B            Mark current frame as BOTTOM
  U            Undo last mark for current droplet
  N            Start a New droplet
  V            Set / change Voltage for current droplet
  +  or  =     Speed up playback  (×1.5)
  -            Slow down playback (÷1.5)
  P            Switch to Previous droplet
  Ctrl+S       Save data to JSON
  G            Go to a specific frame number
  Q  or  Esc   Quit  (auto-saves)

Usage
-----
    python video_player.py                     # interactive file picker
    python video_player.py data/TRIAL_1.mp4    # open specific file
"""

import cv2
import json
import sys
import os
import numpy as np
from pathlib import Path


# ── OpenCV extended key codes (Windows) ──────────────────────────────────────
KEY_ESC        = 27
KEY_SPACE      = 32
KEY_LEFT       = 2424832
KEY_RIGHT      = 2555904
KEY_UP         = 2490368
KEY_DOWN       = 2621440

# Display constants
MAX_DISPLAY_W  = 1280
MAX_DISPLAY_H  = 800
OVERLAY_ALPHA  = 0.55
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL     = cv2.FONT_HERSHEY_PLAIN


class FrameMarker:
    """Interactive video player for marking oil-drop oscillation frames."""

    # ── Initialisation ───────────────────────────────────────────────────────
    def __init__(self, video_path: str):
        self.video_path = os.path.abspath(video_path)
        self.video_name = Path(video_path).stem

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute display scale
        sx = MAX_DISPLAY_W / self.frame_w
        sy = MAX_DISPLAY_H / self.frame_h
        self.scale = min(sx, sy, 1.0)
        self.disp_w = int(self.frame_w * self.scale)
        self.disp_h = int(self.frame_h * self.scale)

        # Playback state
        self.current_frame = 0
        self.playing       = False
        self.speed         = 1.0
        self.frame_img     = None        # current raw frame (full res)

        # Data
        self.droplets: list[dict] = []
        self.current_drop_idx     = -1

        # Output paths
        self.data_dir   = Path(self.video_path).parent
        self.json_path  = self.data_dir / f"{self.video_name}_marks.json"
        self.frames_dir = self.data_dir / "frames" / self.video_name
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Try to load existing session
        self._load()

        # Ensure at least one droplet exists
        if not self.droplets:
            self._new_droplet(prompt_voltage=False)

    # ── Data persistence ─────────────────────────────────────────────────────
    def _save(self):
        data = {
            "video_file":   os.path.basename(self.video_path),
            "fps":          round(self.fps, 4),
            "total_frames": self.total_frames,
            "droplets":     self.droplets,
        }
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVED] {self.json_path}")

    def _load(self):
        if self.json_path.exists():
            with open(self.json_path) as f:
                data = json.load(f)
            self.droplets = data.get("droplets", [])
            if self.droplets:
                self.current_drop_idx = len(self.droplets) - 1
            print(f"[LOADED] {self.json_path}  "
                  f"({len(self.droplets)} droplet(s))")

    # ── Droplet management ───────────────────────────────────────────────────
    def _new_droplet(self, prompt_voltage: bool = True):
        v_lo, v_hi = None, None
        if prompt_voltage:
            v_lo, v_hi = self._prompt_voltage()
        drop = {
            "id":           len(self.droplets) + 1,
            "voltage_lo_V": v_lo,
            "voltage_hi_V": v_hi,
            "marks":        [],
        }
        self.droplets.append(drop)
        self.current_drop_idx = len(self.droplets) - 1
        if v_lo is not None:
            print(f"[NEW DROPLET] #{drop['id']}  "
                  f"V={v_lo}-{v_hi} V")
        else:
            print(f"[NEW DROPLET] #{drop['id']}  V=(not set)")

    @property
    def _drop(self) -> dict | None:
        if 0 <= self.current_drop_idx < len(self.droplets):
            return self.droplets[self.current_drop_idx]
        return None

    # ── Marking ──────────────────────────────────────────────────────────────
    def _mark(self, mark_type: str):
        """Record a TOP or BOTTOM mark at the current frame."""
        drop = self._drop
        if drop is None:
            return
        time_s = self.current_frame / self.fps
        mark = {
            "type":   mark_type,
            "frame":  self.current_frame,
            "time_s": round(time_s, 4),
        }
        drop["marks"].append(mark)

        # Save the frame image
        if self.frame_img is not None:
            fname = (f"drop{drop['id']}_{mark_type}_"
                     f"frame{self.current_frame}.png")
            cv2.imwrite(str(self.frames_dir / fname), self.frame_img)

        print(f"  [{mark_type.upper()}]  frame {self.current_frame}  "
              f"t={time_s:.3f}s  (droplet #{drop['id']})")

    def _undo(self):
        drop = self._drop
        if drop and drop["marks"]:
            removed = drop["marks"].pop()
            # Remove saved frame image
            fname = (f"drop{drop['id']}_{removed['type']}_"
                     f"frame{removed['frame']}.png")
            img_path = self.frames_dir / fname
            if img_path.exists():
                img_path.unlink()
            print(f"  [UNDO] removed {removed['type']} "
                  f"at frame {removed['frame']}")

    # ── Prompt helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _prompt_voltage() -> tuple[float | None, float | None]:
        """Ask the user for the plate voltage low/high (in the terminal)."""
        try:
            lo = input("  Enter V_low  (V) for this droplet "
                       "[Enter to skip]: ").strip()
            hi = input("  Enter V_high (V) for this droplet "
                       "[Enter to skip]: ").strip()
            if lo and hi:
                v_lo, v_hi = float(lo), float(hi)
                if v_lo > v_hi:
                    v_lo, v_hi = v_hi, v_lo
                return v_lo, v_hi
        except (ValueError, EOFError):
            pass
        return None, None

    @staticmethod
    def _prompt_frame() -> int | None:
        try:
            val = input("  Go to frame #: ").strip()
            if val:
                return int(val)
        except (ValueError, EOFError):
            pass
        return None

    # ── Video navigation ─────────────────────────────────────────────────────
    def _seek(self, frame_no: int):
        frame_no = max(0, min(frame_no, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.current_frame = frame_no

    def _read_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if ret:
            self.frame_img = frame.copy()
            self.current_frame = int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            return frame
        return None

    # ── Overlay drawing ──────────────────────────────────────────────────────
    def _draw_overlay(self, display: np.ndarray) -> np.ndarray:
        """Draw semi-transparent information overlay on the display frame."""
        overlay = display.copy()
        h, w = overlay.shape[:2]

        # ── Top-left: frame & time info ──────────────────────────────────
        time_s = self.current_frame / self.fps
        lines_tl = [
            f"Frame: {self.current_frame}/{self.total_frames - 1}",
            f"Time:  {time_s:.2f}s / {self.total_frames / self.fps:.1f}s",
            f"Speed: {self.speed:.2f}x  "
            f"{'>> PLAYING' if self.playing else '|| PAUSED'}",
        ]
        y0 = 20
        for i, txt in enumerate(lines_tl):
            y = y0 + i * 22
            cv2.putText(overlay, txt, (10, y), FONT, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, txt, (10, y), FONT, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)

        # ── Top-right: controls ──────────────────────────────────────────
        controls = [
            "Space:Play/Pause  D/Right:+1  A/Left:-1",
            "W/Up:+10  S/Down:-10  +/-:Speed",
            "T:Top  B:Bottom  U:Undo  N:New Drop",
            "V:Voltage  P:Prev Drop  G:GoTo  Q:Quit",
        ]
        for i, txt in enumerate(controls):
            y = y0 + i * 18
            tw = cv2.getTextSize(txt, FONT_SMALL, 1.0, 1)[0][0]
            cv2.putText(overlay, txt, (w - tw - 10, y), FONT_SMALL, 1.0,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, txt, (w - tw - 10, y), FONT_SMALL, 1.0,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # ── Bottom: current droplet info ─────────────────────────────────
        drop = self._drop
        if drop:
            v_lo = drop.get("voltage_lo_V")
            v_hi = drop.get("voltage_hi_V")
            if v_lo is not None and v_hi is not None:
                v_str = f"{v_lo:.1f}-{v_hi:.1f} V"
            else:
                v_str = "NOT SET"
            info = (f"Droplet #{drop['id']}  |  "
                    f"Voltage: {v_str}  |  "
                    f"Marks: {len(drop['marks'])}")
            cv2.putText(overlay, info, (10, h - 50), FONT, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, info, (10, h - 50), FONT, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)

            # Show last few marks
            recent = drop["marks"][-6:]
            mark_str = "  ".join(
                f"{'T' if m['type'] == 'top' else 'B'}"
                f"@{m['frame']}"
                for m in recent
            )
            if mark_str:
                cv2.putText(overlay, f"Recent: {mark_str}",
                            (10, h - 28), FONT, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, f"Recent: {mark_str}",
                            (10, h - 28), FONT, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # ── Bottom: timeline bar with marks ──────────────────────────────
        bar_y  = h - 12
        bar_h  = 8
        bar_x0 = 10
        bar_x1 = w - 10
        bar_w  = bar_x1 - bar_x0

        # Background bar
        cv2.rectangle(overlay, (bar_x0, bar_y),
                       (bar_x1, bar_y + bar_h), (60, 60, 60), -1)

        # Playback position
        pos_x = bar_x0 + int(
            self.current_frame / max(1, self.total_frames - 1) * bar_w)
        cv2.rectangle(overlay, (bar_x0, bar_y),
                       (pos_x, bar_y + bar_h), (100, 100, 100), -1)

        # Draw all marks from all droplets
        colors_top = (0, 0, 255)     # Red for TOP
        colors_bot = (255, 0, 0)     # Blue for BOTTOM
        for d in self.droplets:
            for m in d["marks"]:
                mx = bar_x0 + int(
                    m["frame"] / max(1, self.total_frames - 1) * bar_w)
                color = colors_top if m["type"] == "top" else colors_bot
                cv2.line(overlay, (mx, bar_y), (mx, bar_y + bar_h),
                         color, 2)

        # Current position indicator
        cv2.circle(overlay, (pos_x, bar_y + bar_h // 2), 5,
                   (0, 255, 255), -1)

        # Blend overlay
        cv2.addWeighted(overlay, OVERLAY_ALPHA, display,
                        1 - OVERLAY_ALPHA, 0, display)
        # Re-draw key text without blending for readability
        for i, txt in enumerate(lines_tl):
            y = y0 + i * 22
            cv2.putText(display, txt, (10, y), FONT, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)

        return display

    # ── Trackbar callback ────────────────────────────────────────────────────
    def _on_trackbar(self, pos: int):
        self._seek(pos)

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        win = "Millikan Oil Drop - Frame Marker"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.disp_w, self.disp_h)
        cv2.createTrackbar("Frame", win, 0,
                           self.total_frames - 1, self._on_trackbar)

        # Read the first frame
        frame = self._read_frame()

        print("\n" + "=" * 60)
        print("  Millikan Oil Drop — Frame Marker")
        print("=" * 60)
        print(f"  Video : {os.path.basename(self.video_path)}")
        print(f"  Frames: {self.total_frames}  |  "
              f"FPS: {self.fps:.2f}  |  "
              f"Duration: {self.total_frames / self.fps:.1f}s")
        print(f"  Output: {self.json_path}")
        print("=" * 60)
        print("  Press 'T' at TOP, 'B' at BOTTOM of each oscillation.")
        print("  Press 'N' to start a new droplet.\n")

        while True:
            if self.playing:
                frame = self._read_frame()
                if frame is None:
                    self.playing = False
                    self._seek(self.total_frames - 1)
                    frame = self._read_frame()
            else:
                # Re-read current frame when paused (for seeking)
                self._seek(self.current_frame)
                frame = self._read_frame()

            if frame is None:
                break

            # Resize for display
            if self.scale < 1.0:
                display = cv2.resize(frame, (self.disp_w, self.disp_h))
            else:
                display = frame.copy()

            # Draw overlay
            display = self._draw_overlay(display)

            # Update trackbar position
            cv2.setTrackbarPos("Frame", win, self.current_frame)

            cv2.imshow(win, display)

            # Wait time depends on playback speed
            if self.playing:
                wait_ms = max(1, int(1000 / (self.fps * self.speed)))
            else:
                wait_ms = 50  # idle polling when paused

            key = cv2.waitKeyEx(wait_ms)

            # ── Handle keys ──────────────────────────────────────────────
            if key == -1:
                continue

            # Quit
            if key in (ord('q'), ord('Q'), KEY_ESC):
                self._save()
                break

            # Play / Pause
            elif key == KEY_SPACE:
                self.playing = not self.playing

            # Step forward 1
            elif key in (ord('d'), ord('D'), KEY_RIGHT):
                self.playing = False
                step = 5 if key == ord('D') else 1
                self._seek(self.current_frame + step)

            # Step backward 1
            elif key in (ord('a'), ord('A'), KEY_LEFT):
                self.playing = False
                step = 5 if key == ord('A') else 1
                self._seek(self.current_frame - step)

            # Step forward 10
            elif key in (ord('w'), ord('W'), KEY_UP):
                self.playing = False
                self._seek(self.current_frame + 10)

            # Step backward 10  (lowercase 's' when paused = step back)
            elif key == KEY_DOWN:
                self.playing = False
                self._seek(self.current_frame - 10)

            # Mark TOP
            elif key in (ord('t'), ord('T')):
                self._mark("top")

            # Mark BOTTOM
            elif key in (ord('b'), ord('B')):
                self._mark("bottom")

            # Undo
            elif key in (ord('u'), ord('U')):
                self._undo()

            # New droplet
            elif key in (ord('n'), ord('N')):
                self.playing = False
                self._new_droplet()

            # Set voltage
            elif key in (ord('v'), ord('V')):
                self.playing = False
                v_lo, v_hi = self._prompt_voltage()
                if v_lo is not None and self._drop:
                    self._drop["voltage_lo_V"] = v_lo
                    self._drop["voltage_hi_V"] = v_hi
                    print(f"  [VOLTAGE] Droplet #{self._drop['id']} "
                          f"-> {v_lo}-{v_hi} V")

            # Previous droplet
            elif key in (ord('p'), ord('P')):
                if self.current_drop_idx > 0:
                    self.current_drop_idx -= 1
                    print(f"  Switched to droplet "
                          f"#{self._drop['id']}")
                elif self.droplets:
                    # Wrap to last
                    self.current_drop_idx = len(self.droplets) - 1
                    print(f"  Switched to droplet "
                          f"#{self._drop['id']}")

            # Speed up
            elif key in (ord('+'), ord('=')):
                self.speed = min(self.speed * 1.5, 16.0)
                print(f"  Speed: {self.speed:.2f}x")

            # Slow down
            elif key == ord('-'):
                self.speed = max(self.speed / 1.5, 0.1)
                print(f"  Speed: {self.speed:.2f}x")

            # Save
            elif key in (ord('s'), ord('S')):
                if not self.playing:
                    # When paused, 's' also steps back 10
                    self._seek(self.current_frame - 10)
                else:
                    self._save()

            # Go to frame
            elif key in (ord('g'), ord('G')):
                self.playing = False
                f = self._prompt_frame()
                if f is not None:
                    self._seek(f)

            # Ctrl+S  (key code 19 on many systems)
            elif key == 19:
                self._save()

        cv2.destroyAllWindows()
        self.cap.release()
        print("\nDone. Goodbye!")


# ── Entry point ──────────────────────────────────────────────────────────────
def pick_video() -> str:
    """Let the user choose a video from the data/ folder."""
    data_dir = Path(__file__).parent / "data"
    videos = sorted(data_dir.glob("*.mp4")) + sorted(data_dir.glob("*.avi"))
    if not videos:
        print("No video files found in data/")
        sys.exit(1)
    print("\nAvailable videos:")
    for i, v in enumerate(videos, 1):
        print(f"  [{i}] {v.name}")
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(videos)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return str(videos[idx])
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = pick_video()

    marker = FrameMarker(video_path)
    marker.run()


if __name__ == "__main__":
    main()

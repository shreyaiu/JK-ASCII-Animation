import cv2
import numpy as np
import subprocess
import shutil
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ---------------- CONFIGURATION ----------------
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "ascii_hacker_animation.mp4"

OUTPUT_FPS = 30
MAX_COLUMNS = 180
CELL_WIDTH = 8
CELL_HEIGHT = 15


# Brightness ramp: dark -> bright
ASCII_CHARS = ".;JUNGKOOKjungook:♡"

# Subtle cyan-white terminal glow
CYAN_TINT = (190, 235, 255)  # B, G, R
GLOW_STRENGTH = 0.45
GLOW_BLUR = 2.0
# ------------------------------------------------


def load_monospace_font(size=12):
    """Find a monospace font on common Windows/Linux installations."""
    candidates = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_glyph_atlas(chars, font, cell_w, cell_h):
    """Render each ASCII glyph once so frames can be built efficiently."""
    atlas = np.zeros((len(chars), cell_h, cell_w), dtype=np.uint8)
    for i, char in enumerate(chars):
        image = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(image)
        draw.text((0, -2), char, font=font, fill=255)
        atlas[i] = np.asarray(image)
    return atlas


def play_video_in_window(video_path, fps):
    """Loop a video in a window; Space pauses or resumes, Q or Esc exits."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for playback: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine video dimensions: {video_path}")

    window_name = "ASCII Video Animation"
    paused = False
    frame_delay = max(1, round(1000 / fps))
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read video for playback: {video_path}"
                    )
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
            if key == ord(" "):
                paused = not paused
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def main():
    source = Path(INPUT_VIDEO)
    if not source.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_VIDEO!r}. Put the video beside this script "
            "or change INPUT_VIDEO to its full path."
        )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Preserve source aspect ratio. A dense character grid represents each frame.
    columns = min(MAX_COLUMNS, max(100, width // CELL_WIDTH))
    rows = max(
        1,
        round(columns * height / width * CELL_WIDTH / CELL_HEIGHT)
    )

    font = load_monospace_font(CELL_HEIGHT)
    chars = "".join(dict.fromkeys(ASCII_CHARS))
    atlas = make_glyph_atlas(chars, font, CELL_WIDTH, CELL_HEIGHT)

    temp_video = Path("ascii_video_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(temp_video), fourcc, OUTPUT_FPS, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the temporary video.")

    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Convert actual source frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Reduce the image to the character grid.
        small = cv2.resize(
            gray, (columns, rows), interpolation=cv2.INTER_AREA
        )

        # 3. Improve contrast while keeping dark regions nearly black.
        brightness = np.clip(
            (small.astype(np.float32) - 12) * 1.18, 0, 255
        ).astype(np.uint8)

        # 4. Map each brightness sample to an actual ASCII glyph.
        indices = (
            brightness.astype(np.uint16) * (len(chars) - 1) // 256
        ).astype(np.uint8)

        glyphs = atlas[indices]
        glyph_mask = glyphs.transpose(0, 2, 1, 3).reshape(
            rows * CELL_HEIGHT, columns * CELL_WIDTH
        )

        # 5. Tint the glyphs cyan-white according to source brightness.
        intensity = cv2.resize(
            brightness, (columns * CELL_WIDTH, rows * CELL_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.float32) / 255.0

        mask = glyph_mask.astype(np.float32) / 255.0
        luminance = np.clip(mask * (0.38 + 0.62 * intensity), 0, 1)

        # OpenCV uses BGR color order.
        b, g, r = CYAN_TINT
        colored = np.stack(
            [luminance * b, luminance * g, luminance * r], axis=-1
        )
        colored = np.clip(colored, 0, 255).astype(np.uint8)

        # 6. Add a restrained glow behind the sharp character shapes.
        glow = cv2.GaussianBlur(
            colored, (0, 0), sigmaX=GLOW_BLUR, sigmaY=GLOW_BLUR
        )
        rendered = np.clip(
            colored.astype(np.float32)
            + glow.astype(np.float32) * GLOW_STRENGTH,
            0, 255
        ).astype(np.uint8)

        # 7. Place the character image on a pure black canvas.
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        rendered_h, rendered_w = rendered.shape[:2]
        x = max(0, (width - rendered_w) // 2)
        y = max(0, (height - rendered_h) // 2)

        copy_h = min(rendered_h, height - y)
        copy_w = min(rendered_w, width - x)
        canvas[y:y + copy_h, x:x + copy_w] = rendered[:copy_h, :copy_w]

        writer.write(canvas)
        frame_number += 1

        if frame_number % 30 == 0:
            if total_frames:
                print(f"Processed {frame_number}/{total_frames} frames")
            else:
                print(f"Processed {frame_number} frames")

    cap.release()
    writer.release()

    # Re-encode as H.264 and copy original audio when FFmpeg is available.
    playback_video = temp_video
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", str(source),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-vf", f"fps={OUTPUT_FPS}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            OUTPUT_VIDEO,
        ]
        subprocess.run(command, check=True)
        playback_video = Path(OUTPUT_VIDEO)
        print(f"\nDone! Saved: {playback_video.resolve()}")
    else:
        print(
            "\nFFmpeg was not found, so the final MP4 with original audio "
            f"was not created. Playing the silent intermediate: {temp_video.resolve()}"
        )

    print(f"Source FPS: {source_fps:.2f}; output FPS: {OUTPUT_FPS}")
    print(f"ASCII grid: {columns} columns × {rows} rows")
    print(f"Processed frames: {frame_number}")
    print("Playing the ASCII animation in a window. Press Q or Esc to close.")
    play_video_in_window(playback_video, OUTPUT_FPS)


if __name__ == "__main__":
    main()
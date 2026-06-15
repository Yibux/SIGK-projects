from pathlib import Path

from PIL import Image, ImageSequence


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
GIF_FILES = [
    OUTPUT_DIR / "walk_generated.gif",
    OUTPUT_DIR / "jump_generated.gif",
]


def zoom_frame(frame: Image.Image, zoom: float) -> Image.Image:
    if zoom <= 1:
        return frame.copy()

    width, height = frame.size
    crop_width = round(width / zoom)
    crop_height = round(height / zoom)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = frame.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def slow_down_gif(path: Path, fps: int = 4, zoom: float = 1.35) -> None:
    duration_ms = round(1000 / fps)

    with Image.open(path) as image:
        frames = [
            zoom_frame(frame.convert("RGBA"), zoom)
            for frame in ImageSequence.Iterator(image)
        ]

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    for path in GIF_FILES:
        if not path.exists():
            print(f"Pominieto brakujacy plik: {path}")
            continue

        slow_down_gif(path)
        print(f"Spowolniono animacje: {path}")


if __name__ == "__main__":
    main()

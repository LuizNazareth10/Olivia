from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:/Olivia")
SOURCE_DIR = ROOT / "img_article"
FIGURES_DIR = ROOT / "Artigo_SBBD_extracted" / "figures"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:/Windows/Fonts/arialbd.ttf",
                r"C:/Windows/Fonts/segoeuib.ttf",
                r"C:/Windows/Fonts/calibrib.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                r"C:/Windows/Fonts/arial.ttf",
                r"C:/Windows/Fonts/segoeui.ttf",
                r"C:/Windows/Fonts/calibri.ttf",
            ]
        )

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int, int], radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def add_text_block(
    image: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font_size: int,
    bold: bool = False,
    bg: tuple[int, int, int, int],
    fg: tuple[int, int, int, int],
    padding: int = 18,
    line_spacing: int = 8,
    radius: int = 18,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    rounded_box(draw, box, bg, radius=radius)
    font = load_font(font_size, bold=bold)
    x1, y1, x2, y2 = box
    x = x1 + padding
    y = y1 + padding
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fg)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing


def translate_app_image() -> None:
    source = Image.open(SOURCE_DIR / "app.jpg").convert("RGBA")
    image = source.crop((0, 0, 760, 1280)).copy()

    light_bg = (252, 252, 253, 252)
    white_bg = (255, 255, 255, 252)
    dark_text = (28, 32, 45, 255)
    muted_text = (105, 111, 129, 255)

    overlays = [
        ((18, 20, 470, 176), "Hello, Admin User", 44, True, light_bg, dark_text),
        ((22, 102, 408, 176), "How are you feeling today?", 20, False, light_bg, muted_text),
        ((24, 184, 696, 308), "Stress Trend", 34, True, white_bg, dark_text),
        ((16, 642, 420, 744), "Quick Actions", 32, True, light_bg, dark_text),
        ((44, 838, 302, 1040), "Start\nCapture", 28, True, (28, 190, 139, 250), (255, 255, 255, 255)),
        ((382, 838, 670, 1040), "View\nHistory", 28, True, (100, 96, 231, 250), (255, 255, 255, 255)),
        ((56, 1140, 185, 1236), "Home", 22, False, light_bg, (108, 82, 223, 255)),
        ((242, 1140, 402, 1236), "History", 22, False, light_bg, muted_text),
        ((430, 1140, 550, 1236), "Tips", 22, False, light_bg, muted_text),
        ((606, 1140, 742, 1236), "Profile", 22, False, light_bg, muted_text),
    ]

    for box, text, size, bold, bg, fg in overlays:
        add_text_block(image, box, text, font_size=size, bold=bold, bg=bg, fg=fg)

    image.convert("RGB").save(FIGURES_DIR / "app.jpg", quality=95)


def translate_control_center() -> None:
    image = Image.open(SOURCE_DIR / "Control Center.jpeg").convert("RGBA")

    panel_bg = (21, 28, 45, 238)
    card_bg = (24, 31, 48, 240)
    dark_bg = (9, 13, 27, 238)
    title_text = (245, 246, 250, 255)
    body_text = (169, 176, 194, 255)

    overlays = [
        ((230, 72, 685, 144), "Simulation & Training", 26, True, panel_bg, title_text),
        (
            (230, 112, 1025, 196),
            "Run full pipeline: Synthetic data generation (5 profiles)\nLocal training -> Aggregation -> Global training.",
            18,
            False,
            panel_bg,
            body_text,
        ),
        ((1270, 76, 1528, 138), "CURRENT STATUS", 18, True, card_bg, body_text),
        ((243, 343, 536, 406), "Personal Models", 22, True, card_bg, title_text),
        ((933, 343, 1220, 406), "Global Model", 22, True, card_bg, title_text),
        ((945, 408, 1048, 452), "ROUND", 14, True, card_bg, body_text),
        ((1050, 408, 1170, 452), "CLIENTS", 14, True, card_bg, body_text),
        ((1184, 408, 1306, 452), "SAMPLES", 14, True, card_bg, body_text),
        ((1388, 408, 1466, 452), "MAE", 14, True, card_bg, body_text),
        ((1514, 408, 1590, 452), "RMSE", 14, True, card_bg, body_text),
    ]

    for box, text, size, bold, bg, fg in overlays:
        add_text_block(image, box, text, font_size=size, bold=bold, bg=bg, fg=fg, padding=14, radius=12)

    add_text_block(
        image,
        (250, 164, 680, 228),
        "Run Full Simulation Pipeline",
        font_size=15,
        bold=True,
        bg=(250, 249, 248, 244),
        fg=(37, 39, 53, 255),
        padding=12,
        radius=14,
    )
    add_text_block(
        image,
        (744, 164, 972, 228),
        "Check Cloud Coordinator",
        font_size=15,
        bold=True,
        bg=(28, 33, 49, 244),
        fg=(226, 228, 234, 255),
        padding=12,
        radius=14,
    )

    draw = ImageDraw.Draw(image, "RGBA")
    rounded_box(draw, (0, 0, 1600, 52), dark_bg, radius=0)
    title_font = load_font(18, bold=True)
    draw.text((236, 15), "Fog Control Center", font=title_font, fill=title_text)

    image.convert("RGB").save(FIGURES_DIR / "control_center.jpg", quality=95)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    translate_app_image()
    translate_control_center()
    print("Translated images written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
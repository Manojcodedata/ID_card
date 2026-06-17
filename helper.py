import cv2
import numpy as np
import urllib.request
import os
from PIL import Image, ImageDraw, ImageFont


# LOAD HAAR CASCADE

HAAR_FACE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)

if face_cascade.empty():
    raise IOError("Failed to load Haar Cascade for face detection")


# FACE DETECTION + CROP 
def detect_and_crop_face(img, padding_factor=2.0):
    """
    Detects the largest face and returns:
    - image (cropped face OR original image)
    - face_found (True / False)
    """

    if img is None:
        return img, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    # No face detected → fallback
    if len(faces) == 0:
        return img, False

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    cx = x + w // 2
    cy = y + h // 2

    crop_size = int(max(w, h) * padding_factor)
    half = crop_size // 2

    h_img, w_img = img.shape[:2]

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w_img)
    y2 = min(cy + half, h_img)

    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return img, False

    return cropped, True

# to paste the tag
def paste_tag(background, overlay, x, y):

    h, w = overlay.shape[:2]

    # Prevent overflow
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    roi = background[y:y+h, x:x+w]

    alpha = alpha[:, :, np.newaxis]

    blended = (roi * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)

    background[y:y+h, x:x+w] = blended

    return background

# ============================================================
# 1. IMAGE ALIGNER (PASSPORT CIRCLE)
# ============================================================
def img_aligner(pic,template_path,tag_path):

    card = cv2.imread(template_path)
    tag = cv2.imread(tag_path, cv2.IMREAD_UNCHANGED)

    if card is None:
        raise FileNotFoundError("student_id.jpg not found")

    center = (319, 345)
    radius = 154
    cx, cy = center

    passport = cv2.resize(pic, (2 * radius, 2 * radius))

    mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    passport_circle = cv2.bitwise_and(passport, passport, mask=mask)

    roi = card[cy - radius:cy + radius, cx - radius:cx + radius]
    background = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    final = cv2.add(background, passport_circle)
    card[cy - radius:cy + radius, cx - radius:cx + radius] = final

    tag_cropped = tag_extract(tag)

    id_image = paste_tag(card, tag_cropped, x=385,y=405)

    return id_image



# GOOGLE DRIVE → DIRECT LINK
def convert_to_direct(url):
    if "open?id=" in url:
        return "https://drive.google.com/uc?id=" + url.split("open?id=")[1]
    return url


# LOAD IMAGE FROM URL
def load_image(url):
    try:
        if not url or str(url).strip() == "":
            return None

        req = urllib.request.urlopen(url, timeout=10)
        data = req.read()

        if data is None or len(data) == 0:
            return None

        arr = np.asarray(bytearray(data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print(f"[ERROR] Image load failed: {e}")
        return None

# TAG EXTRACT
def tag_extract(tag_img):
    # If image has alpha channel
    if tag_img.shape[2] == 4:
        alpha = tag_img[:, :, 3]

        coords = cv2.findNonZero(alpha)
        x, y, w, h = cv2.boundingRect(coords)

        cropped = tag_img[y:y+h, x:x+w]
        return cropped

    # If image has no alpha channel, remove white background
    elif tag_img.shape[2] == 3:
        b, g, r = cv2.split(tag_img)

        # create mask for non-white pixels
        mask = cv2.inRange(tag_img, (0, 0, 0), (245, 245, 245))

        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)

        cropped_bgr = tag_img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        # Convert BGR to BGRA
        cropped_bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
        cropped_bgra[:, :, 3] = cropped_mask

        return cropped_bgra


# FONT RESIZE (FOR STUDENT NAME)
def fit_font_size(text, font_path, max_width, max_size, min_size):
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            return font

    return ImageFont.truetype(font_path, min_size)



# FIT THE TEXT AT LEFT
def draw_left_middle(draw, text, x, center_y, font, fill):
    draw.text((x, center_y), text, font=font, fill=fill, anchor="lm")

# FIT TEXT IN THE MIDDLE
def draw_center_middle(draw, text, center_x, center_y, font, fill):
    draw.text((center_x, center_y), text, font=font, fill=fill, anchor="mm")

# DS ID TEXT ALIGNER
def generate_ds_card(
    id_img,
    student_name,
    enrollment_id,
    father_mother_name,
    course_opted,
    enrollment_validity,
    branch_name,
    poppins_bold_path,
    poppins_regular_path
):
    
    img = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    black = (0, 0, 0)

    name_font = fit_font_size(
        student_name.upper(),
        poppins_bold_path,
        max_width=470,
        max_size=40,
        min_size=24
    )

    enrollment_font = ImageFont.truetype(poppins_bold_path, 30)
    regular_font = ImageFont.truetype(poppins_regular_path, 27)

    # Student Name
    draw_center_middle(
        draw,
        student_name.upper(),
        center_x=320,
        center_y=585,
        font=name_font,
        fill=black
    )

    # Enrollment ID
    draw_left_middle(
        draw,
        enrollment_id,
        x=315,
        center_y=640,
        font=enrollment_font,
        fill=black
    )

    # Father/Mother Name
    draw_left_middle(
        draw,
        father_mother_name,
        x=258,
        center_y=687,
        font=regular_font,
        fill=black
    )

    # Course Opted
    draw_left_middle(
        draw,
        course_opted,
        x=270,
        center_y=730,
        font=regular_font,
        fill=black
    )

    # Enrollment Validity
    draw_left_middle(
        draw,
        enrollment_validity,
        x=330,
        center_y=775,
        font=regular_font,
        fill=black
    )

    # Branch Name
    draw_left_middle(
        draw,
        branch_name,
        x=270,
        center_y=820,
        font=regular_font,
        fill=black
    )

    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return final_img

# DA CARD TEXT ALIGNER
def generate_da_card(
    id_img,
    student_name,
    enrollment_id,
    father_mother_name,
    course_opted,
    enrollment_validity,
    branch_name,
    poppins_bold_path,
    poppins_regular_path
):
    img = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    black = (0, 0, 0)

    name_font = fit_font_size(
        student_name.upper(),
        poppins_bold_path,
        max_width=470,
        max_size=40,
        min_size=24
    )

    enrollment_font = ImageFont.truetype(poppins_bold_path, 30)
    regular_font = ImageFont.truetype(poppins_regular_path, 27)

    # Student Name
    draw_center_middle(
        draw,
        student_name.upper(),
        center_x=320,
        center_y=585,
        font=name_font,
        fill=black
    )

    # Enrollment ID
    draw_left_middle(
        draw,
        enrollment_id,
        x=285,
        center_y=653,
        font=enrollment_font,
        fill=black
    )

    # Father/Mother Name
    draw_left_middle(
        draw,
        father_mother_name,
        x=265,
        center_y=695,
        font=regular_font,
        fill=black
    )

    # Course Opted
    draw_left_middle(
        draw,
        course_opted,
        x=275,
        center_y=738,
        font=regular_font,
        fill=black
    )

    # Enrollment Validity
    draw_left_middle(
        draw,
        enrollment_validity,
        x=337,
        center_y=783,
        font=regular_font,
        fill=black
    )

    # Branch Name
    draw_left_middle(
        draw,
        branch_name,
        x=276,
        center_y=829,
        font=regular_font,
        fill=black
    )

    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return final_img

# FSD CARD TEXT ALIGNER
def generate_fsd_card(
    id_img,
    student_name,
    enrollment_id,
    father_mother_name,
    course_opted,
    enrollment_validity,
    branch_name,
    poppins_bold_path,
    poppins_regular_path
):
    img = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    black = (0, 0, 0)

    name_font = fit_font_size(
        student_name.upper(),
        poppins_bold_path,
        max_width=470,
        max_size=40,
        min_size=24
    )

    enrollment_font = ImageFont.truetype(poppins_bold_path, 30)
    regular_font = ImageFont.truetype(poppins_regular_path, 27)

    # Student Name
    draw_center_middle(
        draw,
        student_name.upper(),
        center_x=320,
        center_y=635,
        font=name_font,
        fill=black
    )

    # Enrollment ID
    draw_left_middle(
        draw,
        enrollment_id,
        x=285,
        center_y=695,
        font=enrollment_font,
        fill=black
    )

    # Father/Mother Name
    draw_left_middle(
        draw,
        father_mother_name,
        x=265,
        center_y=738,
        font=regular_font,
        fill=black
    )

    # Course Opted
    draw_left_middle(
        draw,
        course_opted,
        x=275,
        center_y=783,
        font=regular_font,
        fill=black
    )

    # Enrollment Validity
    draw_left_middle(
        draw,
        enrollment_validity,
        x=337,
        center_y=827,
        font=regular_font,
        fill=black
    )

    # Branch Name
    draw_left_middle(
        draw,
        branch_name,
        x=276,
        center_y=874,
        font=regular_font,
        fill=black
    )

    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return final_img


def write_face_log(enroll_id, face_found, log_file="face_detection_log.txt"):
    status = "FACE DETECTED" if face_found else "FACE NOT DETECTED"

    with open(log_file, "a") as f:
        f.write(f"{enroll_id} : {status}\n")
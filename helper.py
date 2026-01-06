import cv2
import numpy as np
import urllib.request
import os

# ============================================================
# 0. LOAD HAAR CASCADE
# ============================================================
HAAR_FACE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)

if face_cascade.empty():
    raise IOError("Failed to load Haar Cascade for face detection")


# ============================================================
# 0.1 FACE DETECTION + SMART CROP 
# ============================================================
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


# ============================================================
# 1. IMAGE ALIGNER (PASSPORT CIRCLE)
# ============================================================
def img_aligner(pic):
    template_path = "student_id.jpg"
    card = cv2.imread(template_path)

    if card is None:
        raise FileNotFoundError("student_id.jpg not found")

    center = (319, 337)
    radius = 152
    cx, cy = center

    passport = cv2.resize(pic, (2 * radius, 2 * radius))

    mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    passport_circle = cv2.bitwise_and(passport, passport, mask=mask)

    roi = card[cy - radius:cy + radius, cx - radius:cx + radius]
    background = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    final = cv2.add(background, passport_circle)
    card[cy - radius:cy + radius, cx - radius:cx + radius] = final

    return card


# ============================================================
# 2. GOOGLE DRIVE → DIRECT LINK
# ============================================================
def convert_to_direct(url):
    if "open?id=" in url:
        return "https://drive.google.com/uc?id=" + url.split("open?id=")[1]
    return url


# ============================================================
# 3. LOAD IMAGE FROM URL
# ============================================================
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



# ============================================================
# 4. CENTERED TEXT (AUTO-FIT)
# ============================================================
def put_centered_autofit_text(
    img,
    text,
    center_x,
    center_y,
    max_width,
    font,
    max_scale,
    min_scale,
    color=(0, 0, 0),
    thickness=2
):
    if not text:
        return

    text = str(text).strip()
    if text == "":
        return

    scale = max_scale

    while scale >= min_scale:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        if w <= max_width:
            break
        scale -= 0.02

    x = int(center_x - w / 2)
    y = int(center_y + h / 2)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 5. TEXT ALIGNER (ALL FIELDS CENTERED)
# ============================================================
def text_aligner(img, name, enroll_id, father, course, valid):
    CENTER_X_student = 319
    MAX_WIDTH_student = 520
    CENTER_X_father = 450
    MAX_WIDTH_father = 500
    CENTER_X = 319
    MAX_WIDTH = 520
    COLOR = (0, 0, 0)

    FONT_NAME = cv2.FONT_HERSHEY_DUPLEX
    FONT_OTHER = cv2.FONT_HERSHEY_SIMPLEX

    put_centered_autofit_text(img, name, CENTER_X_student, 740, MAX_WIDTH_student,
                              cv2.FONT_HERSHEY_DUPLEX, 1.50, 0.6, (0,0,0))

    cv2.putText(img,father,(270, 872),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(img,enroll_id,(315, 802),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,course,(280, 915),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(img,valid,(336, 966),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),1,cv2.LINE_AA)

    # put_centered_autofit_text(img, enroll_id, 425, 794, MAX_WIDTH,
    #                           FONT_NAME, 1, 0.6, (0,0,255))

    # put_centered_autofit_text(img, father, 400, 860, MAX_WIDTH,
    #                           FONT_NAME, 1, 0.6, COLOR)
    

    # put_centered_autofit_text(img, course, 375, 908, MAX_WIDTH,
    #                           FONT_NAME, 0.8, 0.5, COLOR)

    # put_centered_autofit_text(img, valid, 400, 958, MAX_WIDTH,
    #                           FONT_NAME, 0.7, 0.5, COLOR)

    return img


def write_face_log(enroll_id, face_found, log_file="face_detection_log.txt"):
    status = "FACE DETECTED" if face_found else "FACE NOT DETECTED"

    with open(log_file, "a") as f:
        f.write(f"{enroll_id} : {status}\n")



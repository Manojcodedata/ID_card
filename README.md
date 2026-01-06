# id_card_generation

This repo contains the application code to automate the "Student ID Card" generation.


# Excel → Automated ID Card Generator

---

## 1. Purpose

This application generates **student ID cards in bulk** from an **Excel / CSV file**.
Each student’s photo is processed, aligned onto a fixed ID template, and exported as an image.
Face detection is used when possible, with a **safe fallback** if no face is detected.

---

## 2. High-Level Flow

```
Upload Excel / CSV
 → Read student data
 → Download student photo
 → Face detection + smart crop
 → Passport-style circular alignment
 → Text overlay on ID template
 → Save ID card
 → Log face detection result
 → ZIP download
```

---

## 3. Tech Stack

* **Python**
* **OpenCV** – image processing & face detection
* **Streamlit** – web UI
* **Pandas** – Excel/CSV handling
* **NumPy** – image operations

---

## 4. File Responsibilities

### `helper.py`

Contains all **image processing and utility logic**:

* Face detection and cropping
* Image alignment on ID template
* Image downloading
* Text rendering
* Face detection logging

### `app.py`

Contains the **Streamlit application**:

* File upload UI
* Batch processing loop
* ID card generation
* ZIP file creation

---

## 5. Face Detection Logic (Key Design)

```python
face_img, face_found = detect_and_crop_face(img)
```

* Uses **Haar Cascade** to detect faces
* Detects the **largest face**
* If a face is found → cropped face is used
* If no face is found → **original image is passed forward**

---

## 6. Image Alignment

```python
id_card = img_aligner(face_img)
```

* Resizes input image
* Applies circular mask (passport-style)
* Blends image into a fixed template (`student_id.jpg`)
* Works identically for both face-cropped and full images

---

## 7. Text Rendering

* Student details are written onto the ID card
* Name is auto-scaled to fit available width
* Other fields are placed at fixed template positions

---

## 8. Face Detection Logging

A **single log file** is generated per run:

```
face_detection_log.txt
```

Each student contributes one entry:

```
EnrollmentID : FACE DETECTED
EnrollmentID : FACE NOT DETECTED
```

---

## 9. Required Excel Columns

| Column Name                 |
| --------------------------- |
| Latest Photo of the Student |
| Student Name                |
| Enrollment ID               |
| Father/Mother Name          |
| Course Opted                |
| Validity                    |

---

## 10. Output

```
generated_images/
 ├── <EnrollmentID>.jpg
 ├── <EnrollmentID>.jpg

face_detection_log.txt
generated_images.zip
```

---

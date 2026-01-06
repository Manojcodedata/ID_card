import streamlit as st
import pandas as pd
import os
import cv2
import zipfile
from helper import (
    convert_to_direct,
    load_image,
    detect_and_crop_face,
    img_aligner,
    text_aligner,
    write_face_log
)

# ------------------------
# Log File
# ------------------------
log_file = "face_detection_log.txt"

# Clear old logs before new run
open(log_file, "w").close()


# ---------------------------
# Helpers
# ---------------------------
def process_excel(uploaded_file, output_folder="generated_images"):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    os.makedirs(output_folder, exist_ok=True)

    log_file = "face_detection_log.txt"
    open(log_file, "w").close()


    for i, link in enumerate(df.get("Latest Photo of the Student", [])):
        download_link = convert_to_direct(link)
        img = load_image(download_link)

        if img is None:
            st.warning(f"Row {i}: image not loaded")
            continue

        # face detection + smart crop
        # face_img = detect_and_crop_face(img)
        face_img, face_found = detect_and_crop_face(img)


        # Existing pipeline remains unchanged
        id_card = img_aligner(face_img)

        name = str(df.get("Student Name", [""] * len(df))[i]).strip()
        enroll_id = str(df.get("Enrollment ID", ["unnamed"] * len(df))[i])
        father = df.get("Father/Mother Name", [""] * len(df))[i]
        course = df.get("Course Opted", [""] * len(df))[i]
        valid = df.get("Validity", [""] * len(df))[i]

        student_img = text_aligner(
            id_card, name, enroll_id, father, course, valid
        )
        write_face_log(enroll_id, face_found, log_file)

        save_path = os.path.join(output_folder, f"{enroll_id}.jpg")
        cv2.imwrite(save_path, student_img)

    return output_folder


def zip_folder(folder_path, zip_path="generated_images.zip"):
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, folder_path)
                zf.write(full, arcname)

    return zip_path


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Excel → ID Card Generator")

uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file:
    if st.button("Generate ID Cards"):
        with st.spinner("Generating..."):
            output_folder = process_excel(uploaded_file)
            zip_path = zip_folder(output_folder)

        st.success("ID cards generated successfully!")

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download ZIP",
                f,
                file_name="generated_images.zip",
                mime="application/zip"
            )

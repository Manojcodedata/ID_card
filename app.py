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
    generate_da_card,
    generate_ds_card,
    generate_fsd_card,
    write_face_log
)
from urls import *

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

    failed_images = []

    for i, link in enumerate(df.get("Latest Photo of the Student", [])):
        download_link = convert_to_direct(link)
        img = load_image(download_link)

        name = str(df.get("Student Name", [""] * len(df))[i]).strip()
        enroll_id = str(df.get("Enrollment ID", ["unnamed"] * len(df))[i])
        father = df.get("Father/Mother Name", [""] * len(df))[i]
        course = df.get("Course Opted", [""] * len(df))[i]
        valid = df.get("Validity", [""] * len(df))[i]
        branch = df.get("Branch", [""] * len(df))[i]

        if img is None:
            failed_images.append({
                "Row Number": i + 2,
                "Student Name": name,
                "Enrollment ID": enroll_id,
                "Original Image Link": link,
                "Direct Image Link": download_link,
                "Reason": "Image not loaded"
            })
            st.warning(f"Row {i + 2}: image not loaded")
            continue

        # face detection + smart crop
        # face_img = detect_and_crop_face(img)
        face_img, face_found = detect_and_crop_face(img)


        # Existing pipeline remains unchanged

        #name = str(df.get("Student Name", [""] * len(df))[i]).strip()
        #enroll_id = str(df.get("Enrollment ID", ["unnamed"] * len(df))[i])
        #father = df.get("Father/Mother Name", [""] * len(df))[i]
        #course = df.get("Course Opted", [""] * len(df))[i]
        #valid = df.get("Validity", [""] * len(df))[i]
        #branch = df.get("Branch", [""] * len(df))[i]

        if course == 'Data Analytics':
            imgx = img_aligner(face_img,da_template,da_tag)
            id_img = generate_da_card(imgx,name,enroll_id,father,course,valid,branch,poppins_bold_path,poppins_semibold_path)
        elif course == 'Data Science':
            imgx = img_aligner(face_img,ds_template,ds_tag)
            id_img = generate_ds_card(imgx,name,enroll_id,father,course,valid,branch,poppins_bold_path,poppins_semibold_path)
        elif course == 'Full Stack Development':
            imgx = img_aligner(face_img,fsd_template,fsd_tag)
            id_img = generate_fsd_card(imgx,name,enroll_id,father,course,valid,branch,poppins_bold_path,poppins_semibold_path)

        write_face_log(enroll_id, face_found, log_file)

        save_path = os.path.join(output_folder, f"{enroll_id}.jpg")
        cv2.imwrite(save_path, id_img)
        if len(failed_images)>0:
            failed_df = pd.DataFrame(failed_images)
            failed_file = "failed_images.csv"
            failed_df.to_csv(failed_file, index=False)
        else:
            failed_file = None

    return output_folder,failed_file


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



if "zip_path" not in st.session_state:
    st.session_state.zip_path = None

if "failed_file" not in st.session_state:
    st.session_state.failed_file = None

if uploaded_file:
    if st.button("Generate ID Cards"):
        with st.spinner("Generating..."):
            output_folder, failed_file = process_excel(uploaded_file)
            zip_path = zip_folder(output_folder)

        st.session_state.zip_path = zip_path
        st.session_state.failed_file = failed_file

        st.success("ID cards generated successfully!")

if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
    with open(st.session_state.zip_path, "rb") as f:
        st.download_button(
            "Download ZIP",
            f,
            file_name="generated_images.zip",
            mime="application/zip",
            key="download_zip"
        )

if st.session_state.failed_file and os.path.exists(st.session_state.failed_file):
    failed_df = pd.read_csv(st.session_state.failed_file)

    if len(failed_df) > 0:
        st.warning(f"{len(failed_df)} image(s) failed to load.")

        with open(st.session_state.failed_file, "rb") as f:
            st.download_button(
                "Download Failed Images CSV",
                f,
                file_name="failed_images.csv",
                mime="text/csv",
                key="download_failed_csv"
            )
    else:
        st.info("All images loaded successfully.")

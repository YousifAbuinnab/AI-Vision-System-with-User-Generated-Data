# AI Vision System with User-Generated Data

An end-to-end AI web application where users upload images, the system classifies each image, generates a depth-like map, stores metadata in SQLite, and visualizes analytics in a dashboard.

## Course Connection

This project connects with the following course topics:

- **User Generated Data**: Images uploaded by users are collected and tracked.
- **Machine and Pattern Classification**: Uses a pretrained `ResNet18` model for image classification.
- **Seminar in AI: FoundationStereo / Zero-Shot Stereo Matching**: Includes a stereo-inspired depth estimation module (OpenCV-based placeholder for lightweight depth visualization).
- **Computational Data Analytics**: Uses SQLite + pandas + matplotlib to produce dashboard metrics and trends.

## Features

- Upload JPG/PNG images in a Streamlit UI.
- Save uploaded images to `data/uploads/`.
- Classify images using a pretrained torchvision model (top 3 classes + confidence scores).
- Generate and display a depth-like map.
- Save depth outputs to `data/depth_outputs/`.
- Store metadata in SQLite automatically:
  - image filename
  - upload time
  - predicted class
  - confidence
  - depth output path
- Analytics dashboard with:
  - total uploaded images
  - most common predicted classes
  - average confidence
  - uploads over time

## Project Structure

```text
ai-vision-user-generated-data/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- database.py
|-- models/
|   |-- classifier.py
|   \-- depth_estimator.py
|-- utils/
|   |-- image_utils.py
|   \-- analytics.py
|-- data/
|   |-- uploads/
|   \-- depth_outputs/
\-- assets/
    \-- demo_images/
```

## Installation

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. (Optional) Create a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Example Screenshots (Placeholders)

- Upload and analysis page screenshot:
  - `assets/demo_images/upload_page_example.png`
- Dashboard screenshot:
  - `assets/demo_images/dashboard_example.png`

Add your own screenshots to `assets/demo_images/` as you test the app.

## Future Improvements

- Replace placeholder depth module with a real MiDaS or stereo depth model.
- Add user authentication and role-based data access.
- Support batch image uploads.
- Add model version tracking for reproducible experiments.
- Export analytics reports as CSV/PDF.
- Deploy the app on Streamlit Community Cloud or a containerized platform.

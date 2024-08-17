# 📸 Object Detection and Measurement with TensorFlow and OpenCV

Welcome to the **Object Detection and Measurement** project! This repository contains a Python-based solution leveraging advanced technologies like TensorFlow, OpenCV, and NumPy to detect objects in images or videos and measure their dimensions in real-time. This tool can be used in various applications, such as automated inspection systems, robotics, and more!

## 🧰 Technologies Used

This project utilizes the following technologies:

- **Python** 🐍: The programming language used to develop the project.
- **OpenCV (`cv2`)** 📷: An open-source computer vision and machine learning software library used for image and video processing.
- **NumPy (`np`)** 🔢: A library for numerical computations in Python, used here for mathematical operations and array manipulations.
- **TensorFlow (`tf`)** 🧠: An open-source machine learning framework used to load and run the pre-trained object detection model.
- **TensorFlow's Keras Utility (`get_file`)** 💾: A utility from Keras to download pre-trained models.
- **Computer Vision Techniques** 👁️: Employed to identify and process images for object detection.
- **Object Detection (Pre-trained Model)** 🎯: A deep learning model used to identify objects within images or video frames.
- **Real-Time Video Processing** 🎥: Processing video streams in real-time to detect and measure objects.

## 🚀 Features

- **Object Detection**: Detect objects within images or video streams using a pre-trained TensorFlow model.
- **Dimension Measurement**: Automatically measure the width and height of detected objects in centimeters.
- **Real-Time Processing**: Process video streams in real-time, making it suitable for applications like live monitoring and automated inspections.
- **Customizable Threshold**: Set custom thresholds for detection and non-max suppression to fine-tune detection performance.

## 🗂️ Project Structure

Here's a brief overview of the main files in this project:

- `detector.py `👁️: The main script containing the `Detector` class, which includes methods for reading classes, downloading the model, loading the model, and processing images or videos.
- `Helper.py `👤: A helper script containing utility functions for image warping, contour detection, and point reordering.
- `requirements.txt `📋: A file listing the required Python libraries to run the project.

## 📝 How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/object-detection-measurement.git
    cd object-detection-measurement
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and load the pre-trained model**:
    ```python
    from detector import Detector
    
    detector = Detector()
    detector.download('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz')
    detector.loadModel()
    ```

4. **Run the detection on an image**:
    ```python
    detector.predictImage('path_to_your_image.jpg')
    ```

5. **Or run detection on a video**:
    ```python
    detector.predectVideo('path_to_your_video.mp4')
    ```

## 🖼️ Example Output

<img src="/image.png" width="200" height="200">

## ⚙️ Customization

You can customize the following parameters:

- **Threshold**: Adjust the detection confidence threshold.
- **Non-Max Suppression (NMS)**: Modify the NMS threshold to fine-tune detection accuracy.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions, improvements, or bug reports.

---

✨ Happy coding! ✨

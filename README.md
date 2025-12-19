# 🚗 Autonomous Driving System  
### Lane Detection + Object Detection using YOLOv8 & OpenCV

This project is a mini autonomous-driving perception system built using **Python**, **OpenCV**, and **YOLOv8**.  
It processes road videos to detect:

- **Lane lines** (left and right)
- **Objects on the road** such as cars, trucks, buses, bikes, and pedestrians

The output is a processed video where lane lines and detected objects are overlaid on each frame.

---

## 🎥 Demo

### Lane Detection + Object Detection (Highway)
![Demo 1](demo.gif)

### Urban Road Scenario
![Demo 2](demo2.gif)

### Multi-Lane Road Scenario
![Demo 3](demo3.gif)

▶️ Full demo videos are available locally or via external links (not stored in GitHub due to size limits).

---

## 🔍 Features

### Lane Detection
- HLS color filtering (white & yellow lanes)
- Canny edge detection
- Region of interest masking
- Hough Line Transform
- Temporal smoothing to reduce jitter and false lane jumps

### Object Detection
- YOLOv8 (Ultralytics)
- Detects vehicles and pedestrians
- Real-time capable
- Bounding boxes with confidence scores

### Multi-Video Processing
- Automatically processes all `.mp4` files placed in the `videos/` folder
- Saves results to the `outputs/` folder

## 📁 Project Structure

autonomous-driving/
│
├── lane_and_objects.py # Main script
├── videos/ # Input road videos (local only)
├── outputs/ # Processed videos (generated locally)
├── demo.gif # Short demo shown in README
├── README.md
└── .gitignore

yaml
Copy code

> Note: Large video files are excluded from GitHub and should be kept locally.

---

## ⚙️ Installation

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
YOLOv8 weights will download automatically on first run.

▶️ How to Run
Place your road videos (.mp4) in the videos/ folder

Run the script:

bash
Copy code
python lane_and_objects.py
Processed videos will be saved to the outputs/ folder

Press Q to stop playback

🧠 What I Learned
Classical computer vision techniques for lane detection

Integrating deep learning models with OpenCV pipelines

Temporal smoothing for stable video predictions

Real-time video frame processing

End-to-end ML system structuring

🚀 Future Improvements
Object tracking (DeepSORT)

Lane curvature estimation

Steering angle prediction

Semantic road segmentation (deep learning)

👩‍💻 Author
Haripriya Kotapati
GitHub: https://github.com/kotapatiharipriya


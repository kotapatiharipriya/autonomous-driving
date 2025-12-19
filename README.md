# 🚗 Autonomous Driving System  
### Lane Detection + Object Detection using YOLOv8 & OpenCV

This project is a mini autonomous-driving perception system built using **Python**, **OpenCV**, and **YOLOv8**.  
It processes road videos to detect:

- **Lane lines** (left and right)
- **Objects on the road** such as cars, trucks, buses, bikes, and pedestrians

The output is a processed video where lane lines and detected objects are overlaid on each frame.

---

## 🎥 Demo

![Autonomous Driving Demo](demo.gif)

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

---

## 📁 Project Structure


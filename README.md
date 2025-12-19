# 🚗 Autonomous Driving System  
### Lane Detection + Object Detection Using YOLOv8 & OpenCV

This project is a mini autonomous-driving perception pipeline that I built using **Python, OpenCV, and YOLOv8**.  
It takes road videos and detects two things:

- **Lane lines** (left and right)  
- **Objects on the road** such as cars, buses, trucks, motorbikes, bicycles, and pedestrians  

The output is a processed video where both lane lines and detected objects are drawn on each frame.

---

## 🔥 Features

### ✅ Lane Detection  
- Color filtering (yellow + white lanes)  
- Canny edge detection  
- Region of interest masking  
- Hough line detection  
- Smoothing across frames (reduces shaking)

### ✅ Object Detection (YOLOv8)  
- Detects cars, trucks, buses, bikes, people  
- Fast, accurate, real-time capable  
- Bounding boxes + confidence scores

### ✅ Multi-video Support  
Any number of `.mp4` files inside the `videos/` folder will be processed automatically and saved to `outputs/`.

---

## 📁 Project Structure


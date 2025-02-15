# Advanced Robotic Arm Simulator - Hand Gesture Control

## Overview
This project simulates a robotic arm controlled by hand gestures using OpenCV, MediaPipe, and Pygame. The arm follows the user's hand movements in real-time using inverse kinematics.

## Features
- **Hand Gesture Detection**: Uses MediaPipe Hands to track hand movements.
- **Robotic Arm Simulation**: A three-jointed robotic arm controlled via inverse kinematics.
- **Real-time Interaction**: The arm moves smoothly and naturally in response to gestures.
- **Gripper Control**: Detects pinch gestures to simulate gripping.
- **Metallic Design**: Graphical enhancements using Pygame for realistic rendering.

## Dependencies
Ensure you have the following installed:
```bash
pip install opencv-python mediapipe pygame numpy
```

## Installation & Setup
### 1. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```
### 2. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```
### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```
### 4. Run the Project
```bash
python main.py
```

## Usage
- Place your hand in front of the camera.
- Move your index finger to control the robotic arm.
- Pinch with your thumb and index finger to activate the gripper.
- Close the window or press `Ctrl+C` to exit.

## Files Structure
```
📂 Project Directory
 ├── main.py            # Main script
 ├── requirements.txt   # Dependencies list
 ├── README.md          # Documentation
```

## Troubleshooting
- **Camera Not Detected**: Ensure your webcam is properly connected.
- **Slow Performance**: Reduce `WIDTH` and `HEIGHT` values in the script.
- **Hand Not Recognized**: Ensure proper lighting and keep your hand visible.

## Future Improvements
- Add multiple hand tracking support.
- Improve AI prediction for smoother movements.
- Implement robotic arm hardware integration.

## License
This project is open-source. Feel free to modify and use it as needed.

---
💡 **Tip**: Experiment with different gestures and refine the inverse kinematics logic for better performance!

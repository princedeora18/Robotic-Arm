import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
from pygame import gfxdraw

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1024, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Robotic Arm Simulator - Hand Gesture Control")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 200)
DARK_GRAY = (50, 50, 50)

# Arm properties
base_x, base_y = WIDTH // 2, HEIGHT - 150
arm_length1 = 160
arm_length2 = 120
arm_length3 = 80  # Added third segment for more flexibility
angle1 = 90  # Base rotation
angle2 = 45  # First joint
angle3 = 45  # Second joint
gripper_angle = 0
gripper_size = 20
is_gripping = False

# Physics properties
GRAVITY = 9.81
DAMPING = 0.95
MAX_SPEED = 5
velocity1 = 0
velocity2 = 0
velocity3 = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def draw_metallic_circle(surface, x, y, radius, color):
    """Draw a metallic-looking circle with gradient and highlight"""
    # Draw main circle with gradient
    for i in range(radius, 0, -1):
        intensity = int(255 * (i / radius) * 0.7)
        current_color = (min(color[0] + intensity, 255),
                         min(color[1] + intensity, 255),
                         min(color[2] + intensity, 255))
        gfxdraw.filled_circle(surface, int(x), int(y), i, current_color)

    # Add highlight
    highlight_pos = (int(x - radius / 3), int(y - radius / 3))
    gfxdraw.filled_circle(surface, highlight_pos[0], highlight_pos[1],
                          max(2, int(radius / 4)), (255, 255, 255))


def draw_arm_segment(surface, start_pos, end_pos, width, color):
    """Draw a metallic-looking arm segment"""
    direction = np.array([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]])
    length = np.linalg.norm(direction)
    if length == 0:
        return

    normal = np.array([-direction[1], direction[0]]) / length

    # Create polygon points for the arm segment
    points = [
        (start_pos[0] + normal[0] * width, start_pos[1] + normal[1] * width),
        (start_pos[0] - normal[0] * width, start_pos[1] - normal[1] * width),
        (end_pos[0] - normal[0] * width, end_pos[1] - normal[1] * width),
        (end_pos[0] + normal[0] * width, end_pos[1] + normal[1] * width)
    ]

    # Draw main segment
    pygame.draw.polygon(surface, color, points)
    # Add highlight
    highlight_points = [
        (start_pos[0] + normal[0] * width * 0.5, start_pos[1] + normal[1] * width * 0.5),
        (end_pos[0] + normal[0] * width * 0.5, end_pos[1] + normal[1] * width * 0.5)
    ]
    pygame.draw.line(surface, (200, 200, 200), highlight_points[0], highlight_points[1], 2)


def calculate_inverse_kinematics(target_x, target_y):
    """Calculate joint angles using inverse kinematics"""
    # Convert target coordinates to relative to base
    dx = target_x - base_x
    dy = base_y - target_y

    # Calculate distance to target
    distance = math.sqrt(dx ** 2 + dy ** 2)

    # Check if target is reachable
    if distance > arm_length1 + arm_length2:
        return None

    # Calculate angles using cosine law
    try:
        angle1 = math.degrees(math.atan2(dy, dx))
        angle2_cos = (distance ** 2 + arm_length1 ** 2 - arm_length2 ** 2) / (2 * distance * arm_length1)
        angle2 = math.degrees(math.acos(max(-1, min(1, angle2_cos))))
        angle3_cos = (arm_length1 ** 2 + arm_length2 ** 2 - distance ** 2) / (2 * arm_length1 * arm_length2)
        angle3 = 180 - math.degrees(math.acos(max(-1, min(1, angle3_cos))))

        return angle1, angle2, angle3
    except:
        return None


def process_hand_gesture(hand_landmarks):
    """Process hand landmarks and return control values"""
    global is_gripping

    # Get important landmark positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    wrist = hand_landmarks.landmark[0]

    # Calculate hand orientation and gesture features
    hand_direction = math.atan2(wrist.y - middle_tip.y, wrist.x - middle_tip.x)
    pinch_distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    # Detect gripping gesture (pinch)
    is_gripping = pinch_distance < 0.05

    return hand_direction, is_gripping


running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # Capture and process webcam frame
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Process hand gestures
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_direction, is_gripping = process_hand_gesture(hand_landmarks)

            # Update arm angles based on hand position
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            target_x = base_x + (index_tip.x - 0.5) * WIDTH
            target_y = base_y - index_tip.y * HEIGHT

            # Apply inverse kinematics
            ik_result = calculate_inverse_kinematics(target_x, target_y)
            if ik_result:
                target_angle1, target_angle2, target_angle3 = ik_result

                # Apply smooth movement with velocity and damping
                velocity1 = (target_angle1 - angle1) * 0.1
                velocity2 = (target_angle2 - angle2) * 0.1
                velocity3 = (target_angle3 - angle3) * 0.1

                velocity1 *= DAMPING
                velocity2 *= DAMPING
                velocity3 *= DAMPING

                angle1 += np.clip(velocity1, -MAX_SPEED, MAX_SPEED)
                angle2 += np.clip(velocity2, -MAX_SPEED, MAX_SPEED)
                angle3 += np.clip(velocity3, -MAX_SPEED, MAX_SPEED)

    # Calculate joint positions
    rad1 = math.radians(angle1)
    rad2 = math.radians(angle2)
    rad3 = math.radians(angle3)

    joint1_x = base_x + arm_length1 * math.cos(rad1)
    joint1_y = base_y - arm_length1 * math.sin(rad1)

    joint2_x = joint1_x + arm_length2 * math.cos(rad1 + rad2)
    joint2_y = joint1_y - arm_length2 * math.sin(rad1 + rad2)

    end_x = joint2_x + arm_length3 * math.cos(rad1 + rad2 + rad3)
    end_y = joint2_y - arm_length3 * math.sin(rad1 + rad2 + rad3)

    # Draw enhanced robotic arm
    # Base platform
    pygame.draw.rect(screen, DARK_GRAY, (base_x - 50, base_y - 10, 100, 60))
    pygame.draw.rect(screen, GRAY, (base_x - 40, base_y - 5, 80, 50))

    # Arm segments with metallic effect
    draw_arm_segment(screen, (base_x, base_y), (joint1_x, joint1_y), 15, GRAY)
    draw_arm_segment(screen, (joint1_x, joint1_y), (joint2_x, joint2_y), 12, GRAY)
    draw_arm_segment(screen, (joint2_x, joint2_y), (end_x, end_y), 10, GRAY)

    # Joints with metallic effect
    draw_metallic_circle(screen, base_x, base_y, 20, DARK_GRAY)
    draw_metallic_circle(screen, joint1_x, joint1_y, 15, DARK_GRAY)
    draw_metallic_circle(screen, joint2_x, joint2_y, 12, DARK_GRAY)

    # Gripper
    if is_gripping:
        gripper_color = RED
        gripper_size_current = gripper_size * 0.5
    else:
        gripper_color = BLUE
        gripper_size_current = gripper_size

    # Draw gripper prongs
    grip_angle = math.radians(angle1 + angle2 + angle3)
    grip_dx = math.cos(grip_angle + math.pi / 4) * gripper_size_current
    grip_dy = -math.sin(grip_angle + math.pi / 4) * gripper_size_current

    pygame.draw.line(screen, gripper_color, (end_x, end_y),
                     (end_x + grip_dx, end_y + grip_dy), 5)
    pygame.draw.line(screen, gripper_color, (end_x, end_y),
                     (end_x - grip_dx, end_y - grip_dy), 5)

    # Display OpenCV frame
    cv2.imshow("Hand Gesture Control", frame)

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS

cap.release()
cv2.destroyAllWindows()
pygame.quit()
import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
from pygame import gfxdraw

pygame.init()

WIDTH, HEIGHT = 1024, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotic Arm Simulator")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 200)
DARK_GRAY = (50, 50, 50)

base_x, base_y = WIDTH // 2, HEIGHT - 150
arm_length1, arm_length2, arm_length3 = 160, 120, 80
angle1, angle2, angle3 = 90, 45, 45
gripper_angle = 0
gripper_size = 20
is_gripping = False

GRAVITY = 9.81
DAMPING = 0.95
MAX_SPEED = 5
velocity1 = velocity2 = velocity3 = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def draw_metallic_circle(surface, x, y, radius, color):
    for i in range(radius, 0, -1):
        intensity = int(255 * (i / radius) * 0.7)
        gfxdraw.filled_circle(surface, int(x), int(y), i, (min(color[0] + intensity, 255),
                                                            min(color[1] + intensity, 255),
                                                            min(color[2] + intensity, 255)))

def draw_arm_segment(surface, start_pos, end_pos, width, color):
    direction = np.array([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]])
    length = np.linalg.norm(direction)
    if length == 0:
        return
    normal = np.array([-direction[1], direction[0]]) / length
    points = [(start_pos[0] + normal[0] * width, start_pos[1] + normal[1] * width),
              (start_pos[0] - normal[0] * width, start_pos[1] - normal[1] * width),
              (end_pos[0] - normal[0] * width, end_pos[1] - normal[1] * width),
              (end_pos[0] + normal[0] * width, end_pos[1] + normal[1] * width)]
    pygame.draw.polygon(surface, color, points)

def calculate_inverse_kinematics(target_x, target_y):
    dx, dy = target_x - base_x, base_y - target_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance > arm_length1 + arm_length2:
        return None
    try:
        angle1 = math.degrees(math.atan2(dy, dx))
        angle2 = math.degrees(math.acos(max(-1, min(1, (distance ** 2 + arm_length1 ** 2 - arm_length2 ** 2) / (2 * distance * arm_length1)))))
        angle3 = 180 - math.degrees(math.acos(max(-1, min(1, (arm_length1 ** 2 + arm_length2 ** 2 - distance ** 2) / (2 * arm_length1 * arm_length2)))))
        return angle1, angle2, angle3
    except:
        return None

def process_hand_gesture(hand_landmarks):
    global is_gripping
    thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
    wrist, middle_tip = hand_landmarks.landmark[0], hand_landmarks.landmark[12]
    hand_direction = math.atan2(wrist.y - middle_tip.y, wrist.x - middle_tip.x)
    pinch_distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    is_gripping = pinch_distance < 0.05
    return hand_direction, is_gripping

running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_direction, is_gripping = process_hand_gesture(hand_landmarks)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            target_x, target_y = base_x + (index_tip.x - 0.5) * WIDTH, base_y - index_tip.y * HEIGHT
            ik_result = calculate_inverse_kinematics(target_x, target_y)
            if ik_result:
                target_angle1, target_angle2, target_angle3 = ik_result
                velocity1 = (target_angle1 - angle1) * 0.1
                velocity2 = (target_angle2 - angle2) * 0.1
                velocity3 = (target_angle3 - angle3) * 0.1
                velocity1 *= DAMPING
                velocity2 *= DAMPING
                velocity3 *= DAMPING
                angle1 += np.clip(velocity1, -MAX_SPEED, MAX_SPEED)
                angle2 += np.clip(velocity2, -MAX_SPEED, MAX_SPEED)
                angle3 += np.clip(velocity3, -MAX_SPEED, MAX_SPEED)
    rad1, rad2, rad3 = map(math.radians, [angle1, angle2, angle3])
    joint1_x, joint1_y = base_x + arm_length1 * math.cos(rad1), base_y - arm_length1 * math.sin(rad1)
    joint2_x, joint2_y = joint1_x + arm_length2 * math.cos(rad1 + rad2), joint1_y - arm_length2 * math.sin(rad1 + rad2)
    end_x, end_y = joint2_x + arm_length3 * math.cos(rad1 + rad2 + rad3), joint2_y - arm_length3 * math.sin(rad1 + rad2 + rad3)
    pygame.draw.rect(screen, DARK_GRAY, (base_x - 50, base_y - 10, 100, 60))
    pygame.draw.rect(screen, GRAY, (base_x - 40, base_y - 5, 80, 50))
    draw_arm_segment(screen, (base_x, base_y), (joint1_x, joint1_y), 15, GRAY)
    draw_arm_segment(screen, (joint1_x, joint1_y), (joint2_x, joint2_y), 12, GRAY)
    draw_arm_segment(screen, (joint2_x, joint2_y), (end_x, end_y), 10, GRAY)
    draw_metallic_circle(screen, base_x, base_y, 20, DARK_GRAY)
    draw_metallic_circle(screen, joint1_x, joint1_y, 15, DARK_GRAY)
    draw_metallic_circle(screen, joint2_x, joint2_y, 12, DARK_GRAY)
    cv2.imshow("Hand Gesture Control", frame)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()

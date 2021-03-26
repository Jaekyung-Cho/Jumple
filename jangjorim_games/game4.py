import cv2
import random
import math
import numpy as np
import os
from jangjorim_client import resource_path

# constants
threshold = 50
r_hand = 0.5
r_foot = 0.15
velocity = 8

# target color
target_color = {
  "blue" : (255,0,0),
  "green" : (0,255,0),
  "red" : (0,0,255),
  "yellow" : (0,255,255),
  "origin" : (255,255,255)
}

# target body parts
target_part = {
  "nose" : (0,0),
  "shoulder" : (5,6),
  "eye" : (1,2),
  "knee" : (13,14),
  "ankle" : (15,16),
  "hand" : (17,18),
  "foot" : (19,20)
}
part_list = ["nose", "shoulder", "eye", "knee", "ankle", "hand", "foot"]

# part images
# knee
img_rate = 0.8
knee_img = cv2.imread(resource_path("jangjorim_games/image/Frame 158.png"), cv2.IMREAD_UNCHANGED)
knee_img = cv2.resize(knee_img, dsize=(int(101*img_rate),int(101*img_rate)))
knee_img = cv2.flip(knee_img,1)

hand_img = cv2.imread(resource_path("jangjorim_games/image/Frame 159.png"), cv2.IMREAD_UNCHANGED)
hand_img = cv2.resize(hand_img, dsize=(int(101*img_rate),int(101*img_rate)))
hand_img = cv2.flip(hand_img,1)

foot_img = cv2.imread(resource_path("jangjorim_games/image/Frame 160.png"), cv2.IMREAD_UNCHANGED)
foot_img = cv2.resize(foot_img, dsize=(int(101*img_rate),int(101*img_rate)))
foot_img = cv2.flip(foot_img,1)

ankle_img = cv2.imread(resource_path("jangjorim_games/image/Frame 161.png"), cv2.IMREAD_UNCHANGED)
ankle_img = cv2.resize(ankle_img, dsize=(int(101*img_rate),int(101*img_rate)))
ankle_img = cv2.flip(ankle_img,1)

nose_img = cv2.imread(resource_path("jangjorim_games/image/Frame 162.png"), cv2.IMREAD_UNCHANGED)
nose_img = cv2.resize(nose_img, dsize=(int(101*img_rate),int(101*img_rate)))
nose_img = cv2.flip(nose_img,1)

eye_img = cv2.imread(resource_path("jangjorim_games/image/Frame 163.png"), cv2.IMREAD_UNCHANGED)
eye_img = cv2.resize(eye_img, dsize=(int(101*img_rate),int(101*img_rate)))
eye_img = cv2.flip(eye_img,1)

shoulder_img = cv2.imread(resource_path("jangjorim_games/image/Frame 164.png"), cv2.IMREAD_UNCHANGED)
shoulder_img = cv2.resize(shoulder_img, dsize=(int(168*img_rate),int(92*img_rate)))
shoulder_img = cv2.flip(shoulder_img,1)

target_img = {
  "nose" : nose_img,
  "shoulder" : shoulder_img,
  "eye" : eye_img,
  "knee" : knee_img,
  "ankle" : ankle_img,
  "hand" : hand_img,
  "foot" : foot_img
}


# game explanation
explain = "Touch with corresponding body part"

# target shape
star_size = 0.5
star_img = cv2.imread(resource_path("jangjorim_games/image/star.png"), cv2.IMREAD_UNCHANGED)
star_img = cv2.resize(star_img, dsize=(0,0), fx=star_size,fy=star_size)



def distance(hand, target):
  distance = []
  for i in range(2):
    distance.append(math.sqrt((hand[i][0]-target[0])**2 + (hand[i][1]-target[1])**2))
  
  return min(distance[0], distance[1])


def touch_target(img, coord ,target,part, draw_hand):
  touched = False

  # Augmentation
  new_coord = np.zeros((4,2))
  new_coord[0,:] = np.array([(1+r_hand)*coord[9][0]-r_hand*coord[7][0], (1+r_hand)*coord[9][1]-r_hand*coord[7][1]]).astype(np.int32) # left hand
  new_coord[1,:] = np.array([(1+r_hand)*coord[10][0]-r_hand*coord[8][0], (1+r_hand)*coord[10][1]-r_hand*coord[8][1]]).astype(np.int32) # right hand
  new_coord[2,:]= np.array([(1+r_foot)*coord[15][0]-r_foot*coord[13][0], (1+r_foot)*coord[15][1]-r_foot*coord[13][1]]).astype(np.int32) # left foot
  new_coord[3,:] = np.array([(1+r_foot)*coord[16][0]-r_foot*coord[14][0], (1+r_foot)*coord[16][1]-r_foot*coord[14][1]]).astype(np.int32) # right foot

  

  if not target is None:

    pid = target_part[part]
    if pid[0] > 15:
      left_part = (new_coord[pid[0]-17][1].astype(np.int32), new_coord[pid[0]-17][0].astype(np.int32))
      right_part = (new_coord[pid[1]-17][1].astype(np.int32), new_coord[pid[1]-17][0].astype(np.int32))
    else:
      left_part = (coord[pid[0]][1].astype(np.int32), coord[pid[0]][0].astype(np.int32))
      right_part = (coord[pid[1]][1].astype(np.int32), coord[pid[1]][0].astype(np.int32))
    
    if draw_hand:
      img = cv2.circle(img, left_part, radius = 15, color=(0,255,0), thickness=-1)
      img = cv2.circle(img, right_part, radius = 15, color=(0,255,0), thickness=-1)

    if distance([left_part,right_part], target) < threshold:
      target = None
      part = None
      touched = True
      

  return img, target, part, touched



def random_target(img, target, part, target_shape = "circle", color = "yellow"):
  if target is None:
    target = (random.choice([random.randint(25,100),random.randint(img.shape[1]-100,img.shape[1]-25)]),random.randint(200,img.shape[0]-25))
    part = random.choice(part_list)
  else:
    target = (target[0]+random.randint(-velocity,velocity), target[1]+random.randint(-velocity,velocity))
    if(target[1] > img.shape[0] or target[1] < 0 or target[0] > img.shape[1] or target[0] < 0):
      target = None
      part = None
      return img, target, part

  if target_shape is "circle":
    img = cv2.circle(img, target, radius = 30, color=target_color[color], thickness = -1)
  elif target_shape is "star":
    img = image_overlapping(img,star_img, target, target_color[color])
  elif target_shape is "design":
    img = image_overlapping(img, target_img[part], target, target_color[color])

  return img, target, part            



def image_overlapping(screen, img, pos, color):
    x_size, y_size, _ = img.shape
    x1, x2 = pos[1], pos[1]+x_size
    y1, y2 = pos[0], pos[0]+y_size
    if y2 >= screen.shape[1]:
        y2 = screen.shape[1]-1
        y1 = y2 - y_size
    if y1 < 0:
        y1 = 0
        y2 = y1 + y_size
    if x2 >= screen.shape[0]:
        x2 = screen.shape[0]-1
        x1 = x2 - x_size
    if x1 < 0:
        x1 = 0
        x2 = x1 + x_size

    alpha_img = img[:,:,3]/255.0
    alpha_screen = 1- alpha_img

    for c in range(3):
        screen[x1:x2, y1:y2, c] = alpha_img * img[:,:,c]* color[c]/255.0 + alpha_screen * screen[x1:x2, y1:y2, c]

    return screen
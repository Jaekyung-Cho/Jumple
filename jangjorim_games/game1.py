import cv2
import random
import math
import numpy as np
import os
import sys
from jangjorim_client import resource_path


# constants
threshold = 50
interpol = 0.7
velocity = 0

# target color
target_color = {
  "blue" : (255,0,0),
  "green" : (0,255,0),
  "red" : (0,0,255),
  "yellow" : (0,255,255),
  "origin" : (255,255,255)
}

# game explanation
explain = "Catching 5 stars in 10 seconds"

# target shape
img_rate = 0.8
star_img = cv2.imread(resource_path("jangjorim_games/image/Frame 137.png"), cv2.IMREAD_UNCHANGED)
star_img = cv2.resize(star_img, dsize=(int(63*img_rate),int(60*img_rate)))

def distance(hand, target):
  distance = []
  for i in range(2):
    distance.append(math.sqrt((hand[i][0]-target[0])**2 + (hand[i][1]-target[1])**2))
  
  return min(distance[0], distance[1])

def touch_target(img, hand, elbow , target, draw_hand):
  touched = False
  left_hand = (((1+interpol)*hand[0][1]-interpol*elbow[0][1]).astype(np.int32), ((1+interpol)*hand[0][0]-interpol*elbow[0][0]).astype(np.int32))
  right_hand = (((1+interpol)*hand[1][1]-interpol*elbow[1][1]).astype(np.int32), ((1+interpol)*hand[1][0]-interpol*elbow[1][0]).astype(np.int32))
  if draw_hand:
    img = cv2.circle(img, left_hand, radius = 15, color=(0,255,0), thickness=-1)
    img = cv2.circle(img, right_hand, radius = 15, color=(0,255,0), thickness=-1)

  if not target is None:
    if distance([left_hand,right_hand], target) < threshold:
      target = None
      touched = True
      

  return img, target, touched

def random_target(img, target, target_shape = "star", color = "blue"):
  if target is None:
    target = (random.choice([random.randint(25,60),random.randint(img.shape[1]-60,img.shape[1]-25)]),random.randint(100,img.shape[0]-25))
  else:
    target = (target[0], target[1]+ velocity)
    if(target[1] > img.shape[0] - 30):
      target = None
      return img, target

  if target_shape is "circle":
    img = cv2.circle(img, target, radius = 15, color=target_color[color], thickness = -1)
  elif target_shape is "star":
    img = image_overlapping(img,star_img, target, target_color[color])

  return img, target            


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

# def image_overlapping(screen, img, pos, color):
#   x_size, y_size, _ = img.shape
#   x1, x2 = pos[0]-int(x_size/2), pos[0]+x_size-int(x_size/2)
#   y1, y2 = pos[1]-int(y_size/2), pos[1]+y_size-int(y_size/2)

#   alpha_img = img[:,:,3]/255.0
#   alpha_screen = 1- alpha_img

#   for c in range(3):
#     screen[y1:y2, x1:x2, c] = alpha_img * img[:,:,c] * color[c]/255.0 + alpha_screen * screen[y1:y2, x1:x2, c]

#   return screen
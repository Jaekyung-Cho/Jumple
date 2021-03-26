import cv2
import random
import math
import numpy as np
import os
import time
from jangjorim_client import resource_path

# constants
threshold = 50
interpol = 0.7
velocity = 300

# target color
target_color = {
  "blue" : (255,0,0),
  "green" : (0,255,0),
  "red" : (0,0,255),
  "original" : (255,255,255)
}

# game explanation
explain = "Let's play volley ball 1vs1"

# target shape
star_size = 0.4
star_img = cv2.imread(resource_path("jangjorim_games/image/star.png"), cv2.IMREAD_UNCHANGED)
star_img = cv2.resize(star_img, dsize=(0,0), fx=star_size,fy=star_size)

ball_size = 0.45
ball_img_big = cv2.imread(resource_path("jangjorim_games/image/soccerball.png"), cv2.IMREAD_UNCHANGED)
ball_img = cv2.resize(ball_img_big, dsize=(0,0), fx=ball_size,fy=ball_size)

def distance(hand, target):
  distance = []
  for i in range(2):
    distance.append(math.sqrt((hand[i][0]-target[0])**2 + (hand[i][1]-target[1])**2))
  
  return min(distance[0], distance[1])

def touch_target(img, hand, elbow , target, draw_hand):
  left_hand = (((1+interpol)*hand[0][1]-interpol*elbow[0][1]).astype(np.int32), ((1+interpol)*hand[0][0]-interpol*elbow[0][0]).astype(np.int32))
  right_hand = (((1+interpol)*hand[1][1]-interpol*elbow[1][1]).astype(np.int32), ((1+interpol)*hand[1][0]-interpol*elbow[1][0]).astype(np.int32))
  if draw_hand:
    img = cv2.circle(img, left_hand, radius = 15, color=(0,255,0), thickness=-1)
    img = cv2.circle(img, right_hand, radius = 15, color=(0,255,0), thickness=-1)

  score = 0
  if not target is None:
    if not target[2]:
      if distance([left_hand,right_hand], target) < threshold:
        target = (target[0], target[1], True)
        score = 1

  return img, target, score

def random_target(img, target, delta_time, target_shape = "soccerball", color = "original"):
  if target is None:
    target = (0, random.randint(50, img.shape[0]-50), False)
  else:
    if not target[2]: # touched or not
      target = (target[0]+ int(velocity * delta_time), target[1], target[2])
      if(target[0] > img.shape[1]):
        target = None
        return img, target, False
    else:
      target = (target[0]- int(velocity * delta_time), target[1], target[2])
      if(target[0] <=  0):
        target = None
        return img, target, True

  if target_shape is "circle":
    img = cv2.circle(img, (target[0],target[1]), radius = 15, color=target_color[color], thickness = -1)
  elif target_shape is "star":
    img = image_overlapping(img,star_img, target, target_color[color])
  elif target_shape is "soccerball":
    if target[2]:
      img = image_overlapping(img,ball_img_big, target, target_color[color])
    else:
      img = image_overlapping(img,ball_img, target, target_color[color])

  return img, target , None        

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
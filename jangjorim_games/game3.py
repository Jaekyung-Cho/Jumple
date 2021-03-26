import cv2
import random
import math
import numpy as np
import os
from jangjorim_client import resource_path

# constants
threshold = 50
interpol = 0.7
velocity = 15

# target color
target_color = {
  "blue" : (255,0,0),
  "green" : (0,255,0),
  "red" : (0,0,255),
  "original" : (255,255,255)
}

# game explanation
explain = "Try trapping volley ball at red line"

# target shape
star_size = 0.4
star_img = cv2.imread(resource_path("jangjorim_games/image/star.png"), cv2.IMREAD_UNCHANGED)
star_img = cv2.resize(star_img, dsize=(0,0), fx=star_size,fy=star_size)

ball_size = 0.45
ball_img = cv2.imread(resource_path("jangjorim_games/image/soccerball.png"), cv2.IMREAD_UNCHANGED)
ball_img = cv2.resize(ball_img, dsize=(0,0), fx=ball_size,fy=ball_size)

# target shape (volleyball version)
img_rate = 0.8
vball_img = cv2.imread(resource_path("jangjorim_games/image/Frame 152.png"), cv2.IMREAD_UNCHANGED)
vball_img = cv2.resize(vball_img, dsize=(int(75*img_rate),int(76*img_rate)))

def distance(hand, target):
  distance = []
  for i in range(2):
    distance.append(math.sqrt((hand[i][0]-target[0])**2 + (hand[i][1]-target[1])**2))
  
  return min(distance[0], distance[1])

def touch_target(img, hand, elbow , target, hit_range, prev_touch, prev_touch_img, draw_hand):
  touched = False
  touch_img = prev_touch_img
  left_hand = (((1+interpol)*hand[0][1]-interpol*elbow[0][1]).astype(np.int32), ((1+interpol)*hand[0][0]-interpol*elbow[0][0]).astype(np.int32))
  right_hand = (((1+interpol)*hand[1][1]-interpol*elbow[1][1]).astype(np.int32), ((1+interpol)*hand[1][0]-interpol*elbow[1][0]).astype(np.int32))
  if draw_hand:
    img = cv2.circle(img, left_hand, radius = 15, color=(0,255,0), thickness=-1)
    img = cv2.circle(img, right_hand, radius = 15, color=(0,255,0), thickness=-1)

  touch = prev_touch
  score = 0
  if not target is None:
    if distance([left_hand,right_hand], target) < threshold:
      for touched, low, upp, touch_img in hit_range:
        if target[1] > low and target[1] < upp:
          touch = touched
          score = 1
          break
        touch = None
      target = None

  return img, target, touch, score, touch_img

def random_target(img, target, target_shape = "soccerball", color = "original"):
  if target is None:
    target = (random.randint(img.shape[1]-150,img.shape[1]-80), 50, 0)
  else:
    target = (target[0], target[1]+ velocity, target[2]-velocity)
    if(target[1] > img.shape[0] - 50):
      target = None
      return img, target

  if target_shape is "circle":
    img = cv2.circle(img, (target[0],target[1]), radius = 15, color=target_color[color], thickness = -1)
  elif target_shape is "star":
    img = image_overlapping(img,star_img, target, target_color[color])
  elif target_shape is "soccerball":
    img = image_overlapping(img,ball_img, target, target_color[color])
  elif target_shape is "volleyball":
    img = image_overlapping(img,vball_img, target, target_color[color])

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
#   w, h, _ = img.shape
#   M1 = cv2.getRotationMatrix2D((w/2, h/2), pos[2], 1)
#   img = cv2.warpAffine(img, M1, (w,h))
#   img = img[int(w/4):int(3/4*w), int(h/4):int(3/4*h)]

#   x_size, y_size, _ = img.shape
#   x1, x2 = pos[0]-int(x_size/2), pos[0]+x_size-int(x_size/2)
#   y1, y2 = pos[1]-int(y_size/2), pos[1]+y_size-int(y_size/2)

#   alpha_img = img[:,:,3]/255.0
#   alpha_screen = 1- alpha_img

#   for c in range(3):
#     screen[y1:y2, x1:x2, c] = alpha_img * img[:,:,c] * color[c]/255.0 + alpha_screen * screen[y1:y2, x1:x2, c]

#   return screen
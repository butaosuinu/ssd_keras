import pandas as pd
import numpy as np
import copy
import os
import json

def compute_ratio(bb_true, bb_predict):
  """
  bb_true: ground truth bounding box
  bb_predict: predicted bounding box
  bounding box = (x, y, width, height)
  """
  intersection_x = max(min(bb_true[0]+bb_true[2], bb_predict[0]+bb_predict[2]) - max(bb_true[0], bb_predict[0]), 0)
  intersection_y = max(min(bb_true[1]+bb_true[3], bb_predict[1]+bb_predict[3]) - max(bb_true[1], bb_predict[1]), 0)

  area_intersection = intersection_x*intersection_y
  area_true = bb_true[2]*bb_true[3]
  area_predict = bb_predict[2]*bb_predict[3]
  area_union = area_true + area_predict - area_intersection
  
  ratio = area_intersection/float(area_union)

  return ratio


def compute_ap(y_true, y_predict, threshold):
  """
  y_true: list of ground truth bounding box -> [bb_0, bb_1, ...]
  y_predict: list of predicted bounding box sorted in order -> [bb_0, bb_1, ...]
  """
  delrecall = []
  precision = []
  undetected = copy.copy(y_true)
  results = []
  pred_count = 0
  for y_pred in y_predict:
    if len(undetected) > 0:
      pred_count += 1
      #print('--------', pred_count, '-------')
      #print('predicted bb:', y_pred)
      ratios = np.array([(y_t, compute_ratio(y_t, y_pred)) for y_t in undetected])
      #print('score:', ratios[:,1].max())
      detected = ratios[:,1].max() > threshold
      results.append(detected)
      if detected:
        #print('detected')
        delrecall.append(1.0/len(y_true))
        detected_bb = ratios[:,0][ratios[:,1].argmax()]
        #print(detected_bb)
        undetected.remove(detected_bb)
      else:
        #print('undetected')
        delrecall.append(0.0)
      #print('precision so far:', float(np.array(results).sum())/len(results))
      precision.append(float(np.array(results).sum())/len(results))
      #print('remaining', len(undetected), 'bb(s) so far')
      #for u in undetected:
      #    print(u)
      
    else:
      break
  #print('\nDone.\n')
  #print('precision:', precision)
  #print('delrecall:', delrecall)
  ap = np.sum(np.array(precision)*np.array(delrecall))
  #print('average precision:', ap, '\n')
  
  return ap

def compute_map(submit, ans, threshold = 0.9):
  """
  submit: pandas.DataFrame
  ans: pandas.DataFrame
  """
  ans_scores = []
  for a in ans:
    #print(t)
    s_bb = []
    for s in submit:
      if s[0] == a[0] and s[1].upper() == a[1].upper():
        s_bb.append((int(s[3]),int(s[4]),int(s[5]),int(s[6])))

    ans_bb = []
    # ans_true = ans[ans[0] == t]
    for t in ans:
      if a[0] == t[0] and a[1] == t[1]:
        ans_bb.append((int(t[2]),int(t[3]),int(t[4]),int(t[5])))

    # print(ans_bb)
    # print(s_bb)

    ans_scores.append(compute_ap(ans_bb, s_bb, threshold))

  return np.mean(np.array(ans_scores))


sub_path = 'sub_bbs.json'
ans_path = 'ans_bbs.json'

f_sub_bb = open(sub_path, 'r')
f_ans_bb = open(ans_path, 'r')

sub = json.load(f_sub_bb)
ans = json.load(f_ans_bb)

sub_h = []
for s_h in sub:
  if float(s_h[2]) > 0.5:
    print(s_h)
    sub_h.append(s_h)

ans_t = []
sub_t = []
for a in ans:
  if 'title' == a[1]:
    ans_t.append(a)

for s in sub:
  if 'Title' == s[1]:
    sub_t.append(s)

ans_st = []
sub_st = []
for a in ans:
  if 'sub_title' == a[1]:
    ans_st.append(a)

for s in sub:
  if 'Sub_title' == s[1]:
    sub_st.append(s)

ans_at = []
sub_at = []
for a in ans:
  if 'author' == a[1]:
    ans_at.append(a)

for s in sub:
  if 'Author' == s[1]:
    sub_at.append(s)

ans_n = []
sub_n = []
for a in ans:
  if 'num' == a[1]:
    ans_n.append(a)

for s in sub:
  if 'Num' == s[1]:
    sub_n.append(s)

ans_sn = []
sub_sn = []
for a in ans:
  if 'series_name' == a[1]:
    ans_sn.append(a)

for s in sub:
  if 'Series_name' == s[1]:
    sub_sn.append(s)

score = compute_map(sub, ans, 0.6)
score9 = compute_map(sub, ans, 0.9)
scoreh = compute_map(sub_h, ans, 0.6)
scoreh9 = compute_map(sub_h, ans, 0.9)

scoreT = compute_map(sub_t, ans_t, 0.6)
scoreT9 = compute_map(sub_t, ans_t, 0.9)

scoreST = compute_map(sub_st, ans_st, 0.6)
scoreST9 = compute_map(sub_st, ans_st, 0.9)

scoreAT = compute_map(sub_at, ans_at, 0.6)
scoreAT9 = compute_map(sub_at, ans_at, 0.9)

scoreN = compute_map(sub_n, ans_n, 0.6)
scoreN9 = compute_map(sub_n, ans_n, 0.9)

scoreSN = compute_map(sub_sn, ans_sn, 0.6)
scoreSN9 = compute_map(sub_sn, ans_sn, 0.9)

print('信頼性すべて、範囲重なり具合0.6以上')
print(score)
print('信頼性すべて、範囲重なり具合0.9以上')
print(score9)

print('信頼性0.5以上、範囲重なり具合0.6以上')
print(scoreh)
print('信頼性0.5以上、範囲重なり具合0.9以上')
print(scoreh9)

print('title')
print(scoreT)
print(scoreT9)
print('sub_title')
print(scoreST)
print(scoreST9)
print('author')
print(scoreAT)
print(scoreAT9)
print('num')
print(scoreN)
print(scoreN9)
print('series_name')
print(scoreSN)
print(scoreSN9)

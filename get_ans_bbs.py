import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

import pickle

import glob
import json

ans_priors = pickle.load(open('my_test_ans.pkl', 'rb'))

# print(ans_priors)

ans_bb_arr = []
for i, ans in ans_priors.items():
  img_bb = []
  for v in ans:
    # print(v)
    bb = np.r_[['test_img\\' + i], v]
    # bb = np.r_[[i], v]
    img_bb.append(bb.tolist())

  ans_bb_arr.extend(img_bb)

print(ans_bb_arr)
with open('ans_bbs.json', 'w') as fw:
  json.dump(ans_bb_arr, fw, ensure_ascii=False, indent=4)
"""
    加载模型 并使用模型
"""
import numpy as np
import matplotlib.pyplot as mp
import pickle
import sklearn
markerdis = 1.8 #1.8cm边长
x_scale = 1 #pix to cm
y_scale = 1
def pix_scale(x,y):
    global x_scale,y_scale
    # 加载模型 使用模型
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    with open('./linear2_x.pkl', 'rb') as f:
        model = pickle.load(f)
        x_prd_y= model.predict(x)
        x_scale = 1.8 / x_prd_y
    with open('./linear2_y.pkl', 'rb') as f:
        model = pickle.load(f)
        y_prd_y = model.predict(y)
        y_scale = 1.8 / y_prd_y
        #print('ml_scale_use result:','x_scale=', x_scale, 'y_scale=', y_scale)



# Imports
import glob
from tqdm import tqdm 
import random
import cv2
import time
import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from DataGenerator import DataGenerator
try:
    import queue
except ImportError:
    import Queue as queue
import warnings
warnings.filterwarnings("ignore")


class Detection_Pipeline:
    __generator = DataGenerator()
    
    def __sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]
        
    def __intersection(self, g, p):
        g = Polygon(g[:8].reshape((4, 2)))
        p = Polygon(p[:8].reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter/union

    def __weighted_merge(self, g, p):
        g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
        g[8] = (g[8] + p[8])
        return g


    def __standard_nms(self, S, thres):
        order = np.argsort(S[:, 8])[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ovr = np.array([self.__intersection(S[i], S[t]) for t in order[1:]])

            inds = np.where(ovr <= thres)[0]
            order = order[inds+1]

        return S[keep]


    def __nms_locality(self,polys, thres=0.3):
        '''Given poly coordinates and prob score value, it gives boxes after nms'''
        S = []
        p = None

        for g in polys:
            if p is not None and self.__intersection(g, p) > thres:

                p = self.__weighted_merge(g, p)
            else:
                if p is not None:
                    S.append(p)
                p = g

        if p is not None:
            S.append(p)

        if len(S) == 0:
            return np.array([])

        return self.__standard_nms(np.array(S), thres)
    
    def detect(self, img, model):

        #1.Text Detection
        img=cv2.resize(img,(512,512))
        pred_im=model.predict(np.expand_dims(img,axis=0))
        score_map=pred_im[0][:,:,0]
        geo_map=pred_im[0][:,:,1:]

        for i in [0,1,2,3,4]:
            geo_map[:,:,i]*=score_map

        #2.ROI Rotate  
        score_map_thresh=0.5
        box_thresh=0.1 
        nms_thres=0.2
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, :]

        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)

        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        # restore
        text_box_restored = self.__generator.restore_rectangle(xy_text[:, ::-1], geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        boxes = self.__nms_locality(boxes.astype(np.float64), nms_thres)


        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32), 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
            if i==4:
                break
                
        if len(boxes)>0:
            boxes = boxes[boxes[:, 8] > box_thresh]
        boxes[:,:8:2] = np.clip(boxes[:,:8:2], 0, 512 - 1)
        boxes[:,1:8:2] = np.clip(boxes[:,1:8:2], 0, 512 - 1)  
        res = []
        result = []
        
        if len(boxes)>0:
            for box in boxes:
                box_ =  box[:8].reshape((4, 2))
                if np.linalg.norm(box_[0] - box_[1]) < 8 or np.linalg.norm(box_[3]-box_[0]) < 8:
                    continue
                result.append(box_)
        res.append(np.array(result, np.float32))   

        box_index = []
        brotateParas = []
        filter_bsharedFeatures = []
        for i in range(len(res)):
            rotateParas = []
            rboxes=res[i]
            txt=[]
            for j, rbox in enumerate(rboxes):
                para = self.__generator.restore_roiRotatePara(rbox)
                if para and min(para[1][2:]) > 8:
                    rotateParas.append(para)
                    box_index.append((i, j))
            pts=[]   


        #3. Text Recognition (From boxes given by Text Detection+ROI Rotate) 

        if len(rotateParas) > 0:
            for num in range(len(rotateParas)):
                text=""
                out=rotateParas[num][0]
                crop=rotateParas[num][1]
                points=np.array([[out[0],out[1]],[out[0]+out[2],out[1]],[out[0]+out[2],out[1]+out[3]],[out[0],out[1]+out[3]]])
                pts.append(points)

        # 4. Labeling detected and Recognized Text in Image

        for i in range(len(pts)):
            cv2.polylines(img,[pts[i]],isClosed=True,color=(0,255,0),thickness=1)
        return img    


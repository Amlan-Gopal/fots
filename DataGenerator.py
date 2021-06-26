# Imports
import numpy as np
import os
import scipy.io
import glob
from tqdm import tqdm 
import random
import pandas as pd
import pickle
import cv2
import time
import scipy.optimize
from shapely.geometry import Polygon
import tensorflow as tf
import math
import traceback
import warnings
warnings.filterwarnings("ignore")

#get_ipython().system('pip install Shapely')

# References
# https://github.com/jiangxiluning/FOTS.PyTorch/tree/master/FOTS
# https://github.com/Masao-Taketani/FOTS_OCR

class DataGenerator:
    
    def __loadSynthImages(self):
        """Returns SynthText directory files"""
        
        files = []
        for root,dirs,file in os.walk('SampleSynth',topdown=True):
            for i in file:
                extn = i.split('.')[-1]
                if extn != 'DS_Store' and extn!= 'mat':
                    path = root+os.sep+i
                    path = '/'.join(path.split('\\'))
                    files.append(path)
        return files        
    
    def __loadICDARImages(self):
        """Returns ICDAR directory files"""
        
        files = []
        root = 'ch4_training_images'
        for f in os.listdir(root):
            files.append(root + '/' + f)
        return files
    
    def __loadAllSynthGT(self):
        """Loads gt.mat (Ground Truth values) and returns dictinary containing image path, 
        bounding box cordinates and contained text"""
        
        with open('SampleSynth/synth_gt.txt', 'rb') as f:
            data = f.read()
        synth_gt = pickle.loads(data)
        
        return synth_gt
    
    def __loadGT_Synth(self, imgPath, synth_dict= {}):
        """For a given Synth image path, it returns the bounding box cordinates and texts"""
        
        # Check the dictinary
        if len(synth_dict) == 0:
            raise Exception('Synth GT is empty!!')
        else:
            # Collect bounding box cordinates and texts for a given index
            if imgPath in synth_dict['imname']:
                index = synth_dict['imname'].index(imgPath)
                bboxes = synth_dict['wordBB'][index]
                texts = synth_dict['txt'][index]
                _, _, numOfWords = bboxes.shape
                # Change shape from (2, 4, number of boxes) to (number of boxes, 4, 2)
                bboxes = bboxes.reshape([8, numOfWords], order = 'F').T  # num_words * 8
                bboxes = bboxes.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
                texts = np.array([word for line in texts for word in line.split()])
                return bboxes, texts
        return None, None       
                    
    def  __textTagsCounts_ICDAR(self):
        """It will return count of all the texts in the data"""
        textTags = []
        for file in os.listdir('ch4_training_localization_transcription_gt'):
            path = 'ch4_training_localization_transcription_gt' + '/' + file
            
            with open(path, 'r', encoding='utf-8-sig') as f:
                # Split the lines for each bouning box
                lines = f.read().split('\n')
                texts = [''.join(line.split(',')[8:]) for line in lines if line]
                # Check for invalid texts
                textTags.extend([word for line in texts for word in line.split() if not (word == '*' or word == '###')])
         
        return len(set(textTags))
                        
        
    def __loadGT_ICDAR(self, imgPath):
        """For a given ICDAR image path, it returns the bounding box cordinates and texts"""

        # Generate file path from image path
        path=imgPath.split('/')[-1]
        path=path.replace('jpg','txt')
        path=path.replace('png','txt')
        path=path.replace('jpeg','txt')
        path = 'ch4_training_localization_transcription_gt/gt_' + path

        with open(path, 'r', encoding='utf-8-sig') as f:
            # Split the lines for each bouning box
            lines = f.read().split('\n')
            # get bounidng box cordinates
            bboxes = []
            # Get cordinate values and convert them to (4,2) shapes each
            bbox_cords = [line.split(',')[:8] for line in lines if line]
            for bbox_cord in bbox_cords:
                x1, y1, x2, y2, x3, y3, x4, y4 = bbox_cord
                bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                bboxes.append(bbox)
            # Extract text
            texts = [''.join(line.split(',')[8:]) for line in lines if line]
            # Check for invalid texts
            texts = [None if(word == '*' or word == '###') else word for line in texts for word in line.split()]
            return np.array(bboxes, dtype=np.float32), np.array(texts)
        
    def __polygon_area(self, poly):
        """Returns area for polygon coordinates"""
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge)/2

    def __check_and_validate_polys(self, polys, tags, h_w):
        """Given polys and tags with image height width, return the valid polys and coresponding tags"""
        
        (h, w) = h_w
        if polys.shape[0] == 0:
            return polys
        
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        
        for poly, tag in zip(polys, tags):
            p_area = self.__polygon_area(poly)
            
            if abs(p_area) < 1:
                continue
                
            if p_area > 0:
                poly = poly[(0, 3, 2, 1), :]
                
            validated_polys.append(poly)
            validated_tags.append(tag)
        return np.array(validated_polys), np.array(validated_tags)
    
    def __shrink_poly(self, poly, r):
        """Create inner/shrink poly inside given poly for score map"""
        
        R = 0.3 # shrink ratio
        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) >                         np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):

            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)

        return poly
    
    def __point_dist_to_line(self, p1, p2, p3):
        """compute the distance from p3 to p1-p2"""
        
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    
    def __fit_line(self, p1, p2):
        """fit a line ax+by+c=0"""
        
        if p1[0] == p1[1]:
            return [1., 0., -p1[0]]
        else:
            [k, b] = np.polyfit(p1, p2, deg=1)

            return [k, -1., b]
    
    def __line_cross_point(self, line1, line2):
        """Return the cross point of given line1 and line2 (line ax+by+c=0)"""
        
        if line1[0] != 0 and line1[0] == line2[0]:
            print('Cross point does not exist')
            return None
        if line1[0] == 0 and line2[0] == 0:
            print('Cross point does not exist')
            return None
        if line1[1] == 0:
            x = -line1[2]
            y = line2[0] * x + line2[2]
        elif line2[1] == 0:
            x = -line2[2]
            y = line1[0] * x + line1[2]
        else:
            k1, _, b1 = line1
            k2, _, b2 = line2
            x = -(b1-b2)/(k1-k2)
            y = k1*x + b1
        return np.array([x, y], dtype=np.float32)
    
    def __line_verticle(self, line, point):
        """get the verticle line from line across point"""
        
        if line[1] == 0:
            verticle = [0, -1, point[1]]
        else:
            if line[0] == 0:
                verticle = [1, 0, -point[0]]
            else:
                verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
        return verticle
    
    def __rectangle_from_parallelogram(self, poly):
        """Create a rectangle from a parallelogram"""

        p0, p1, p2, p3 = poly
        angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
        if angle_p0 < 0.5 * np.pi:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
                # p0 and p2
                ## p0
                p2p3 = self.__fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.__line_verticle(p2p3, p0)

                new_p3 = self.__line_cross_point(p2p3, p2p3_verticle)
                ## p2
                p0p1 = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.__line_verticle(p0p1, p2)

                new_p1 = self.__line_cross_point(p0p1, p0p1_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
            else:
                p1p2 = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.__line_verticle(p1p2, p0)

                new_p1 = self.__line_cross_point(p1p2, p1p2_verticle)
                p0p3 = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.__line_verticle(p0p3, p2)

                new_p3 = self.__line_cross_point(p0p3, p0p3_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
                # p1 and p3
                ## p1
                p2p3 = self.__fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.__line_verticle(p2p3, p1)

                new_p2 = self.__line_cross_point(p2p3, p2p3_verticle)
                ## p3
                p0p1 = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.__line_verticle(p0p1, p3)

                new_p0 = self.__line_cross_point(p0p1, p0p1_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
            else:
                p0p3 = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.__line_verticle(p0p3, p1)

                new_p0 = self.__line_cross_point(p0p3, p0p3_verticle)
                p1p2 = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.__line_verticle(p1p2, p3)

                new_p2 = self.__line_cross_point(p1p2, p1p2_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
            
    def __sort_rectangle(self, poly):
        """sort the four coordinates of the polygon, points in poly should be sorted clockwise"""

        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
            p0_index = np.argmin(np.sum(poly, axis=1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
        else:
            # find the point that sits right to the lowest point
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
            # assert angle > 0
            if angle <= 0:
                print(angle, poly[p_lowest], poly[p_lowest_right])
            if angle/np.pi * 180 > 45:
                #this point is p2
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
            else:
                # this point is p3
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle
    
    def __restore_rectangle_rbox(self, origin, geometry):
        """Resotre rectangle tbox"""
        
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                          np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                          d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                          -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                          -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))
        return np.concatenate([new_p_0, new_p_1])


    def restore_rectangle(self, origin, geometry):
        return self.__restore_rectangle_rbox(origin, geometry)
    
    #These Functions are used to Generate ROI params like out box,crop box & angles that we use to crop text from image
    def __generate_roiRotatePara(self, box, angle, expand_w = 60):
        """Generate all ROI Parameterts"""
        
        p0_rect, p1_rect, p2_rect, p3_rect = box
        cxy = (p0_rect + p2_rect) / 2.
        size = np.array([np.linalg.norm(p0_rect - p1_rect), np.linalg.norm(p0_rect - p3_rect)])
        rrect = np.concatenate([cxy, size])

        box=np.array(box)

        points=np.array(box, dtype=np.int32)
        xmin=np.min(points[:,0])
        xmax=np.max(points[:,0])
        ymin=np.min(points[:,1])
        ymax=np.max(points[:,1])
        bbox = np.array([xmin, ymin, xmax, ymax])
        if np.any(bbox < -expand_w):
            return None

        rrect[:2] -= bbox[:2]
        rrect[:2] -= rrect[2:] / 2
        rrect[2:] += rrect[:2]

        bbox[2:] -= bbox[:2]

        rrect[::2] = np.clip(rrect[::2], 0, bbox[2])
        rrect[1::2] = np.clip(rrect[1::2], 0, bbox[3])
        rrect[2:] -= rrect[:2]

        return bbox.astype(np.int32), rrect.astype(np.int32), - angle

    def restore_roiRotatePara(self, box):
        rectange, rotate_angle = self.__sort_rectangle(box)
        return self.__generate_roiRotatePara(rectange, rotate_angle)
    
    #This function is used to generate geo_map,score_map, training_mask,corp_box,out_box,angle that we use while training model
    def __generate_rbox(self, im_size, polys, tags, num_classes):
        """Genrate score_map and geo_map for image"""
        
        h, w = im_size
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)

        outBoxs = []
        cropBoxs = []
        angles = []
        text_tags = []
        recg_masks = []
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            #print(poly)
            tag = poly_tag[1]
            #print(tag)
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = self.__shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)

            # if geometry == 'RBOX':
            # generate a parallelogram for any combination of two vertices
            fitted_parallelograms = []
            for i in range(4):
                p0 = poly[i]
                p1 = poly[(i + 1) % 4]
                p2 = poly[(i + 2) % 4]
                p3 = poly[(i + 3) % 4]
                edge = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                backward_edge = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                forward_edge = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                if self.__point_dist_to_line(p0, p1, p2) > self.__point_dist_to_line(p0, p1, p3):
                    #  parallel lines through p2
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p2[0]]
                    else:
                        edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
                else:
                    # after p3
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p3[0]]
                    else:
                        edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
                # move forward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p2 = self.__line_cross_point(forward_edge, edge_opposite)
                if self.__point_dist_to_line(p1, new_p2, p0) > self.__point_dist_to_line(p1, new_p2, p3):
                    # across p0
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p0[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
                else:
                    # across p3
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p3[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
                new_p0 = self.__line_cross_point(forward_opposite, edge)
                new_p3 = self.__line_cross_point(forward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
                # or move backward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p3 = self.__line_cross_point(backward_edge, edge_opposite)
                if self.__point_dist_to_line(p0, p3, p1) > self.__point_dist_to_line(p0, p3, p2):
                    # across p1
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p1[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
                else:
                    # across p2
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p2[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
                new_p1 = self.__line_cross_point(backward_opposite, edge)
                new_p2 = self.__line_cross_point(backward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            areas = [Polygon(t).area for t in fitted_parallelograms]
            parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
            # sort thie polygon
            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

            rectange = self.__rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = self.__sort_rectangle(rectange)

            p0_rect, p1_rect, p2_rect, p3_rect = rectange

            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(p0_rect - p3_rect), np.linalg.norm(p1_rect - p2_rect))
            poly_w = min(np.linalg.norm(p0_rect - p1_rect), np.linalg.norm(p2_rect - p3_rect))

            invaild = (min(poly_h, poly_w) < 6) or tag is None or (True and poly_h > poly_w * 2)

            if invaild:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

            if not invaild:
                roiRotatePara = self.__generate_roiRotatePara(rectange, rotate_angle)
                if roiRotatePara:
                    outBox, cropBox, angle = roiRotatePara
                    if min(cropBox[2:]) > 6:
                        w , h = cropBox[2:]
                        textImgW = np.ceil(min(w / float(h) * 32, 256) / 4 /1)
                        #print(tag)
                        if textImgW >= 2 * min(len(tag), 16):  # avoid CTC error
                            outBoxs.append(outBox)
                            cropBoxs.append(cropBox)
                            angles.append(angle)
                            text_tags.append(tag[:16])
                            recg_masks.append(1.)

            for y, x in xy_in_poly:
                point = np.array([x, y], dtype=np.float32)
                # top
                geo_map[y, x, 0] = self.__point_dist_to_line(p0_rect, p1_rect, point) + 3
                # right
                geo_map[y, x, 1] = self.__point_dist_to_line(p1_rect, p2_rect, point) + 3
                # down
                geo_map[y, x, 2] = self.__point_dist_to_line(p2_rect, p3_rect, point) + 3
                # left
                geo_map[y, x, 3] = self.__point_dist_to_line(p3_rect, p0_rect, point) + 3
                # angle
                geo_map[y, x, 4] = rotate_angle
        if len(outBoxs) == 0:
            outBoxs.append([0, 0, 2 * 4, 2 * 4]) # keep extract From sharedConv feature map not zero
            cropBoxs.append([0, 0, 2 * 4, 2 * 4])
            angles.append(0.)
            text_tags.append([num_classes - 2])
            recg_masks.append(0.)

        outBoxs = np.array(outBoxs, np.int32)
        cropBoxs = np.array(cropBoxs, np.int32)
        angles = np.array(angles, np.float32)

        return score_map, geo_map, training_mask, (outBoxs, cropBoxs, angles), text_tags, recg_masks
    
    def generate(self, input_size=512, batch_size=32, isSynth = True):
        """Given batch size and image shape, generate ground truth features for modeling"""
        
        # Get image path list for Synth or ICDAR image
        image_list = []
        synth_dict = None
        num_classes = 0
        
        if isSynth:
            image_list = self.__loadSynthImages()
            synth_dict = self.__loadAllSynthGT()
            num_classes = len([text for texts in synth_dict['txt'] for text in texts])
        else:
            image_list = self.__loadICDARImages()
            num_classes = self.__textTagsCounts_ICDAR()
        
        if not image_list:
            raise Exception('No image available!!')
        
        index = np.arange(0, len(image_list))
        
        while True:
            
            np.random.shuffle(index)
            images = []
            image_fns = []
            score_maps = []
            geo_maps = []
            training_masks = []
            rboxes = []
            tags = []
            recg_masks = []
        
            for i in index:
                
                try:
                    
                    im_fn = image_list[i]       
                    im = cv2.imread(im_fn,cv2.IMREAD_UNCHANGED)
                    if im is None:
                        continue
                        
                    h, w, _ = im.shape

                    if isSynth:
                        text_polys, text_tags = self.__loadGT_Synth(im_fn, synth_dict)
                    else:
                        text_polys, text_tags = self.__loadGT_ICDAR(im_fn)

                    text_polys, text_tags = self.__check_and_validate_polys(text_polys, text_tags, (h, w))

                    #resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(512, 512),interpolation = cv2.INTER_AREA)

                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape

                    score_map, geo_map, training_mask, rbox, text_tags, recg_mask = self.__generate_rbox((new_h, new_w), text_polys, text_tags, num_classes)

                    images.append(im)
                    image_fns.append(im_fn)
                    score_maps.append(score_map[::, ::, np.newaxis].astype(np.float32))
                    geo_maps.append(geo_map[::, ::, :].astype(np.float32))
                    training_masks.append(training_mask[::, ::, np.newaxis].astype(np.float32))

                    rboxes.append(rbox)
                    tags.append(text_tags)
                    recg_masks.append(recg_mask)
                    
                    if len(images) == batch_size:
                        input_images = np.array(images)
                        feature_maps = np.concatenate([np.array(score_maps), np.array(geo_maps), np.array(training_masks)],axis=3)
                        yield (input_images, feature_maps)
                        images = []
                        image_fns = []
                        score_maps = []
                        geo_maps = []
                        training_masks = []
                        rboxes = []
                        tags = []
                        recg_masks = []
                except Exception as e:
                    print(image_list[i])
                    traceback.print_exc()
                    continue

    def generateRecogTextSynth(self, input_size = 512):
        """Generate synth text images for recognition"""
        # Load image paths and gt dict
        image_list = self.__loadSynthImages()
        synth_dict = self.__loadAllSynthGT()
        num_classes = len([text for texts in synth_dict['txt'] for text in texts])
        # sample 5k images out of 10k
        image_list = random.sample(image_list,5000)
        index = np.arange(0, len(image_list))
        np.random.shuffle(index)
        c=0
        paths=[]
        words=[]
        
        if not os.path.exists('synth_word_texts'):
            os.mkdir('synth_word_texts')
        
        for i in index:
            try:
                im_fn = image_list[i]       
                im = cv2.imread(im_fn,cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue

                h, w, _ = im.shape
                text_polys, text_tags = self.__loadGT_Synth(im_fn, synth_dict)
                text_polys, text_tags = self.__check_and_validate_polys(text_polys, text_tags, (h, w))
                
                #resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(512, 512),interpolation = cv2.INTER_AREA)

                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                    
                score_map, geo_map, training_mask, rbox, text_tags, recg_mask = self.__generate_rbox((new_h, new_w), text_polys, text_tags, num_classes)
                outbox, cropbox,angle=rbox
                for i in range(len(outbox)):
                    if(recg_mask[i]!=0):
                        out=outbox[i]
                        crop=cropbox[i]
                        if(im.shape[0]>out[3]+out[1] and im.shape[1]>out[2]+out[0] and out[2]>=0 and out[3]>=0 and out[1]>=0 and out[0]>=0):
                            ang = angle[i]
                            img = tf.image.crop_to_bounding_box(im,out[1],out[0],out[3],out[2])
                            img = tf.keras.preprocessing.image.random_rotation(img,ang*180/np.pi,)
                            
                            if not isinstance(img,np.ndarray):
                                img=img.numpy()
                            
                            img = cv2.resize(img,(128,64),interpolation = cv2.INTER_AREA)
                            img = cv2.detailEnhance(img)
                            c+=1
                            cv2.imwrite('synth_word_texts/word_'+str(c)+'.png',img)
                            paths.append('synth_word_texts/word_'+str(c)+'.png')
                            words.append(text_tags[i])
           
            except Exception as e:
                print(image_list[i])
                import traceback
                traceback.print_exc()
                continue
         
        data=pd.DataFrame({"path":paths,"word":words})
        return data
        
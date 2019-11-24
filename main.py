# -*- coding: utf-8 -*-

"""
Created on Thursday Nov 21 18:56:11 2019

@author: Bas Brussen
@email: b.brussen@digital-twin.nl
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os, os.path


class OffsetTool(object):
    def __init__(self, path, scale_factor):
        img_list = []
        self.scale_factor = scale_factor
        self.get_images(img_list, path, scale_factor)
        self.img_list = img_list
        self.img_result = []
        self.img_hough = []
        
        self.x_result = np.array([]);
        self.y_result = np.array([]);
        
        # Params for blob detection
        params = cv.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 40;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
         
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.8
        params.maxInertiaRatio = 1.2
        
        # Create detector object
        self.detector = cv.SimpleBlobDetector_create(params)
        
        # Find potential blobs/nozzle
        for i in self.img_list:
            self.blob(i, self.img_result)
        
        # Display images one by one
        #self.display_img(self.img_result )
        
        self.save_image(self.img_result, path)
        
        
    # Blob detection
    def blob(self, src, result):
        # Create blur, then convert to binary
        kernel = np.ones((6,6),np.float32)/30
        src = cv.filter2D(src, -1, kernel)
        null, src = cv.threshold(src, 65, 255, cv.THRESH_BINARY)
        
        # Detect the nozzle
        keypoints = self.detector.detect(src)
        
        # Initilaze
        height, width = src.shape 
        x = 0
        y = 0
        
        # Catch in case no blob is detected
        try:
            # Process deviations
            x = round(((width / 2 - keypoints[0].pt[0]) * -1), 7)
            y = round((height / 2 - keypoints[0].pt[1]), 7)
            
            # Display deviations
            print(x, y)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(src, 'x: ' + str(x), (390,170), font, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(src, 'y: ' + str(y), (390,185), font, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            
            self.x_result = np.append(self.x_result, [x], axis=None)
            self.y_result = np.append(self.y_result, [y], axis=None)
        except Exception as e:
            print(e )
            x = ''
            y = ''
            pass
        
        # Draw circle around nozzle
        im_with_keypoints = cv.drawKeypoints(src, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw centre lines
        image = cv.line(im_with_keypoints, 
                         (int(width / 2), 0), 
                         (int(width / 2), height), 
                         (0, 0, 255), 
                         1) 
        
        image = cv.line(image,
                        (0, int(height / 2)),
                        (width, int(height / 2)),
                        (0, 0, 255), 
                        1) 
        
        result.append(image)
        
    def hough(self, src):
        #gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(src, 5)
        
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=60, param2=40,
                               minRadius=50, maxRadius=250)
    
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(gray, center, radius, (255, 0, 255), 3)
        
        return gray
        
    # Get images in specified directory
    def get_images(self, img_list, path, scale_ratio):
        path = os.path.dirname(os.path.abspath(__file__)) + path

        for filename in os.listdir(path):
            img = cv.imread(os.path.join(path,filename), 0)
            
            if img is not None:
                # Downscale image for less computational recourses
                resized = cv.resize(img, 
                                    (int(img.shape[1] * scale_ratio), 
                                     int(img.shape[0] * scale_ratio)), 
                                     interpolation = cv.INTER_AREA)
                                    
                img_list.append(resized)
    
    # Display images one by one
    def display_img(self, im_list, interpolation=cv.INTER_CUBIC):
        j = 0
        for i in im_list:
            cv.imshow("image {}".format(j), i)
            cv.waitKey(0)
            cv.destroyAllWindows()
            j += 1
    
    def save_image(self, im_list, path):
        j = 0
        for i in im_list:
            cv.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)) + path + "/result", "img_%i.jpg" % j) , i)
            j += 1
        

    
    def plot_normal(self):
        #hx, hy, _ = plt.hist(self.x_result, bins=5)
        plt.figure(figsize=(12, 8))

        self.x_result = self.x_result / self.scale_factor
        self.y_result = self.x_result / self.scale_factor
        
        try:
            plt.subplot(121)
            plt.title('Normal distribution X [pixels]')
            plt.grid()
            plt.subplot(plt.hist(self.x_result, bins=5))
        except:
            pass
        
        #hx, hy, _ = plt.hist(self.y_result, bins=5)
        
        try:
            plt.subplot(122)
            plt.title('Normal distribution Y [pixels]')
            plt.grid()
            plt.subplot(plt.hist(self.y_result, bins=5))
        except:
            pass
        
        plt.show()

if __name__ == '__main__':
    processor1 = OffsetTool("/images/tool1", 0.10)
    processor2 = OffsetTool("/images/tool2", 0.10)
    
    processor1.plot_normal()
    processor2.plot_normal()
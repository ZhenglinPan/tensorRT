# -*- UFT-8 -*- #

import os
import time
import argparse

import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

WINDOW_NAME = 'TrtYOLODemo'
START_FRAME = 5

# Kalman滤波器初始化
kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵H
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵A
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)


# 解析终端输入字符串
def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    '''description = python3 trt_yolo.py --usb 0 -m yolov4-tiny-288'''
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


# 添加Kalman滤波
def kalman_prediction(measurement):  # measurement is like [[x], [y]]
    global current_measurement, last_measurement, current_prediction, last_prediction
    last_prediction = current_prediction 
    last_measurement = current_measurement 
    current_measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]) 
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()

    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]

    return [int(cpx), int(cpy)]


# 筛选出行人类别
def obj_filter(boxes, confs, clss):
    '''
    boxes: 
    [[ 552   73  708  429]      
    [ 535  154  586  309]
    [   5   20  269  363]           # 注: (left, top), (left+w, top+h) 与输入帧尺度一致
    [ 954  226 1166  343]
    [1219  232 1279  440]
    [ 837  235  975  285]
    [ 901  236  999  304]]
    confs: [0.5630284  0.4677704  0.6801914  0.6645249  0.58727616 0.44328296 0.3335283]
    clss: [0. 0. 2. 2. 2. 2. 2.]    #注:0是人
    '''
    idxs = [i for i, c in enumerate(clss) if c == 0]
    clss = [clss[idx] for idx in idxs]
    boxes = [boxes[idx] for idx in idxs]
    confs = [confs[idx] for idx in idxs]

    boxes = np.array(boxes)

    if len(boxes) == 0: # type detect error
        pass
    else:
        index = np.array([i for i in range(len(boxes))]).reshape(len(boxes), 1)
        boxes = np.hstack((boxes, index))

    return boxes, confs, clss


def find_max_box(boxes):
    boxes = boxes.tolist()
    # calculate areas of boundingboxes
    area_list = []
    for box in boxes:
        box_area = (box[1]-box[3])*(box[0]-box[2])
        area_list.append(box_area)
    area_list.index(sorted(area_list)[-1])
    max_box = boxes[area_list.index(max(area_list))][:-1]
    return np.array(max_box)


def similarity(img1, img2):
    # high similarity at less value
    img1 = cv2.resize(img1, (16, 16))
    img2 = cv2.resize(img2, (16, 16))
    return np.sum(np.square(np.subtract(img1, img2)))


def find_best_box(boxes, img, tpl):
    mat_list = []
    sml_list = []
    for box in boxes:
        mat = img[box[1]:box[3], box[0]:box[2]]
        mat_list.append(mat)
        
        sml = similarity(mat, tpl)
        sml_list.append(sml)
    
    print("sml_list", sml_list)
    best_mat = mat_list[sml_list.index(min(sml_list))]
    best_box = boxes[mat_list.index(best_mat)]

    return np.array(best_box)


# 循环检测主函数
def loop_and_detect(cam, trt_yolo, conf_th, vis):
    full_scrn = False
    frame_cnt = 0
    capture = cv2.VideoCapture("E1.mp4")
    fps = 0.0
    tic = time.time()
    pv_detection = []
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        # frame = cam.read()
        ret, frame = capture.read()
        if frame is None:
            break
        frame_h, frame_w = frame.shape[:-1]

        if frame_cnt > START_FRAME:
            if frame_cnt == START_FRAME + 1:
                matching_area = frame

            boxes, confs, clss = trt_yolo.detect(matching_area, conf_th)
            # print("detect_boxes", boxes)
            boxes, confs, clss = obj_filter(boxes, confs, clss)     # boxes --np.array() [[p, p, p, p, index], ...]

            if frame_cnt > START_FRAME + 1:
                if len(boxes) == 0:
                    print("boxes after_filter is empty!")
                    boxes, confs, clss = pv_detection[:]
                else:
                    for i in range(len(boxes)):
                        boxes[i][0:2] = np.add(boxes[i][0:2], np.array(matching_area_top_left))
                        boxes[i][2:4] = np.add(boxes[i][2:4], np.array(matching_area_top_left))
            # 绘图
            if frame_cnt == START_FRAME + 1:
                img = vis.draw_bboxes(frame, boxes, confs, clss, 0, 0)
            else:
                img = vis.draw_bboxes(frame, boxes, confs, clss, 
                                        matching_area_top_left, matching_area_bottom_right)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            
            # nominate desirable object
            if frame_cnt == START_FRAME + 1:
                print("Select the target you want.")
                nmn = int(cv2.waitKey(0)) - 176
                box = boxes[int(np.argwhere(boxes[:, -1] == int(nmn))), :-1]    # box --np.array() [p, p, p, p]
            else:
                box = find_best_box(boxes, frame, template)

            roi = frame[box[1]:box[3], box[0]:box[2]]
            if frame_cnt == START_FRAME + 1:
                template = roi.copy()

            roi_h, roi_w = roi.shape[:-1]
            if roi_h < int(0.5*frame_h):
                roi_h = int(0.5*frame_h)
            if roi_w < int(0.2*frame_w):
                roi_w = int(0.2*frame_w)

            matching_area_top_left = [0, 0]
            matching_area_bottom_right = [0, 0]
            matching_area_top_left[0] = box[0] - int(0.75*roi_w)
            matching_area_top_left[1] = box[1] - int(0.25*roi_h)

            # apply kalman filter
            if frame_cnt > START_FRAME + 20:
                matching_area_top_left_measurement = matching_area_top_left
                matching_area_top_left = kalman_prediction([[matching_area_top_left[0]], 
                                                            [matching_area_top_left[1]]])
                # print(np.subtract(np.array(matching_area_top_left), np.array(matching_area_top_left_measurement)))
            else:
                dump = kalman_prediction([[matching_area_top_left[0]], [matching_area_top_left[1]]])
            
            matching_area_bottom_right[0] = box[2] + int(0.75*roi_w)
            matching_area_bottom_right[1] = box[3] + int(0.25*roi_h)
            # print("area height", abs(matching_area_top_left[1] - matching_area_bottom_right[1]))
            # print("area width", abs(matching_area_top_left[0] - matching_area_bottom_right[0]))
            # 越界处理
            for i in range(len(matching_area_top_left)):
                if  matching_area_top_left[i] < 0:
                    matching_area_top_left[i] = 0
            if  matching_area_bottom_right[0] > frame.shape[1]:
                matching_area_bottom_right[0] = frame.shape[1]
            if  matching_area_bottom_right[1] > frame.shape[0]:
                matching_area_bottom_right[1] = frame.shape[0]
            # 切片 [高, 宽]
            matching_area = frame[matching_area_top_left[1]:matching_area_bottom_right[1], 
                                matching_area_top_left[0]:matching_area_bottom_right[0]]
            # cv2.imshow("matching_area", matching_area)

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

            pv_detection = [boxes, confs, clss]

        # cv2.waitKey(0)
        key = cv2.waitKey(1)
        if key == 27 or key == ord(' '):  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        frame_cnt += 1
        

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    h, w = get_input_shape(args.model)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    
    loop_and_detect(cam, trt_yolo, conf_th=0.7, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

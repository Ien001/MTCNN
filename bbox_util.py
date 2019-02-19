#!/usr/bin/env python3
import time
import numpy as np
import torch

def generate_bbox(map, reg, scale, threshold):
    stride = 2
    cellsize = 12 # receptive field
    #print('map:', map)
    #print('reg:', reg)
    t_index = np.where(map > threshold)
    #print('t_index:', t_index)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])
    
    dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
    reg = np.array([dx1, dy1, dx2, dy2])
    #print('reg np:', reg)
    # t_index[0]: choose the first column of t_index
    # t_index[1]: choose the second column of t_index
    score = map[t_index[0], t_index[1], 0]
    #print('score:', score)
    #  t_index[1] means column, t_index[1] is the value of x
    #  t_index[0] means row, t_index[0] is the value of y
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                             np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                             np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                             np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                             score,
                             reg,
                             ])
    return boundingbox.T

def refine_bbox(all_boxes):
    bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    boxes = np.vstack([all_boxes[:,0],
               all_boxes[:,1],
               all_boxes[:,2],
               all_boxes[:,3],
               all_boxes[:,4],
              ])
    boxes = boxes.T

    # boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
    align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
    align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
    align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
    align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh
    # refine the boxes
    boxes_align = np.vstack([ align_topx,
                          align_topy,
                          align_bottomx,
                          align_bottomy,
                          all_boxes[:, 4], #score
                          ])
    boxes_align = boxes_align.T
    
    return boxes_align


def main():
    clf_fm = torch.rand(1,1,1,1).cuda()   # classification feature map
    bbox_reg = torch.rand(1,4,1,1).cuda() # bbox regression
    scale = 1.0
    threshold = 0.1

    bbox = generate_bbox(clf_fm[0].cpu(), \
                            bbox_reg.cpu().reshape(1,1,1,4).numpy(), \
                            scale, \
                            threshold)
    print('generate_bbox:\n', bbox)

    refined_bbox = refine_bbox(bbox)
    print('refine_bbox:\n', refined_bbox)




if __name__=="__main__":
    main()
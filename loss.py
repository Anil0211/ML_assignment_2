import torch

def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    # areap = max((x2p - x1p + 1), 0.0) * max((y2p - y1p + 1), 0.0)
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)

    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    # areag = max((x2g - x1g + 1), 0.0) * max((y2g - y1g + 1), 0.0)
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    union = areap + areag - inter
    # if union <= 0:
    #     return 0.0
    return inter / union #if union > 0 else 0.0

def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    # device = output.device
    eps = 1e-6

    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    # box_confidence[i, select, j, k] = 1.0

    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                if gt_mask[i, j, k] > 0:
                    gt = gt_box[i, :, j, k].clone()

                    # [FOR IOU only] Transform gt for IoU calculation
                    gt_for_iou = gt.clone()
                    gt_for_iou[0] = gt_for_iou[0] * grid_size + k * grid_size
                    gt_for_iou[1] = gt_for_iou[1] * grid_size + j * grid_size
                    gt_for_iou[2] = gt_for_iou[2] * image_size
                    gt_for_iou[3] = gt_for_iou[3] * image_size

                    select = 0
                    max_iou = -1  # safe default
                    for b in range(num_boxes):
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    # box_confidence[i, select, j, k] = max(0.0,max_iou)
                    box_confidence[i, select, j, k] = 1.0
                    # print('select box %d with iou %.2f' % (select, max_iou))

    weight_coord = 5.0
    weight_noobj = 0.5

    pred_boxes = output[:, :5*num_boxes, :, :].reshape(batch_size, num_boxes, 5, num_grids, num_grids)
    pred_cls = output[:, 5*num_boxes:, :, :]

    loss_x = weight_coord * torch.sum(box_mask * (gt_box[:, 0:1, :, :].unsqueeze(1) - pred_boxes[:, :, 0, :, :]) ** 2)
    loss_y = weight_coord * torch.sum(box_mask * (gt_box[:, 1:2, :, :].unsqueeze(1) - pred_boxes[:, :, 1, :, :]) ** 2)

    loss_w = weight_coord * torch.sum(
        box_mask * (
            torch.sqrt(gt_box[:, 2:3, :, :].unsqueeze(1).clamp(min=eps)) -
            torch.sqrt(pred_boxes[:, :, 2, :, :].clamp(min=eps))
        ) ** 2
    )
    loss_h = weight_coord * torch.sum(
        box_mask * (
            torch.sqrt(gt_box[:, 3:4, :, :].unsqueeze(1).clamp(min=eps)) -
            torch.sqrt(pred_boxes[:, :, 3, :, :].clamp(min=eps))
        ) ** 2
    )


    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
    #loss_noobj = weight_noobj  * torch.sum((1.0-box_mask) * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
    loss_noobj = weight_noobj * torch.sum((1.0 - box_mask) * torch.pow(output[:, 4:5*num_boxes:5], 2.0))

    # loss_cls = torch.sum(gt_mask.unsqueeze(1) * (pred_cls - 1.0) ** 2)  # since it's a one-class problem
    loss_cls = torch.sum(gt_mask.unsqueeze(1) * (pred_cls - 1.0)**2 + (1 - gt_mask.unsqueeze(1)) * (pred_cls)**2)


    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls

    if torch.isnan(loss):
        print("[NaN detected] loss breakdown:")
        print("loss_x:", loss_x.item())
        print("loss_y:", loss_y.item())
        print("loss_w:", loss_w.item())
        print("loss_h:", loss_h.item())
        print("loss_obj:", loss_obj.item())
        print("loss_noobj:", loss_noobj.item())
        print("loss_cls:", loss_cls.item())

    return loss

# """
# CS 6375 Homework 2 Programming
# Implement the compute_loss() function in this python script
# """
# import os
# import torch
# import torch.nn as nn


# # compute Intersection over Union (IoU) of two bounding boxes
# # the input bounding boxes are in (cx, cy, w, h) format
# def compute_iou(pred, gt):
#     x1p = pred[0] - pred[2] * 0.5
#     x2p = pred[0] + pred[2] * 0.5
#     y1p = pred[1] - pred[3] * 0.5
#     y2p = pred[1] + pred[3] * 0.5
#     areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
#     x1g = gt[0] - gt[2] * 0.5
#     x2g = gt[0] + gt[2] * 0.5
#     y1g = gt[1] - gt[3] * 0.5
#     y2g = gt[1] + gt[3] * 0.5
#     areag = (x2g - x1g + 1) * (y2g - y1g + 1)

#     xx1 = max(x1p, x1g)
#     yy1 = max(y1p, y1g)
#     xx2 = min(x2p, x2g)
#     yy2 = min(y2p, y2g)

#     w = max(0.0, xx2 - xx1 + 1)
#     h = max(0.0, yy2 - yy1 + 1)
#     inter = w * h
#     iou = inter / (areap + areag - inter)    
#     return iou

# ## Todo: finish the implementation of this loss function for YOLO training
# # output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# # pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# # gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# # gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# # num_boxes: number of bounding boxes per cell
# # num_classes: number of object classes for detection
# # grid_size: YOLO grid size, 64 in our case
# # image_size: YOLO image size, 448 in our case
# def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
#     batch_size = output.shape[0]
#     num_grids = output.shape[2]
#     # compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
#     box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
#     box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

#     # compute assignment of predicted bounding boxes for ground truth bounding boxes
#     for i in range(batch_size):
#         for j in range(num_grids):
#             for k in range(num_grids):
 
#                 # if the gt mask is 1
#                 if gt_mask[i, j, k] > 0:
#                     # transform gt box
#                     gt = gt_box[i, :, j, k].clone()
#                     gt[0] = gt[0] * grid_size + k * grid_size
#                     gt[1] = gt[1] * grid_size + j * grid_size
#                     gt[2] = gt[2] * image_size
#                     gt[3] = gt[3] * image_size
#                     # print('gt in loss %.2f, %.2f, %.2f, %.2f' % (gt[0], gt[1], gt[2], gt[3]))

#                     select = 0
#                     max_iou = -1
#                     # select the one with maximum IoU
#                     for b in range(num_boxes):
#                         # center x, y and width, height
#                         pred = pred_box[i, 5*b:5*b+4, j, k].clone()
#                         iou = compute_iou(gt, pred)
#                         if iou > max_iou:
#                             max_iou = iou
#                             select = b
#                     box_mask[i, select, j, k] = 1
#                     box_confidence[i, select, j, k] = max_iou
#                     print('select box %d with iou %.2f' % (select, max_iou))

#     # compute yolo loss
#     weight_coord = 5.0
#     weight_noobj = 0.5

#     # according to the YOLO paper, we compute the following losses
#     # loss_x: loss function on x coordinate (cx)
#     # loss_y: loss function on y coordinate (cy)
#     # loss_w: loss function on width 
#     # loss_h: loss function on height
#     # loss_obj: loss function on confidence for objects
#     # loss_nonobj: loss function on confidence for non-objects
#     # loss_cls: loss function for object class

#     # This is implementation for the loss_obj
#     # Follow this example to compute other losses
    
#     pred_boxes = output[:, :5*num_boxes, :, :].reshape(batch_size, num_boxes, 5, num_grids, num_grids)
#     pred_cls = output[:, 5*num_boxes:, :, :]

#     loss_x = weight_coord * torch.sum(box_mask * (gt_box[:, 0:1, :, :].unsqueeze(1) - pred_boxes[:, :, 0, :, :]) ** 2)
#     loss_y = weight_coord * torch.sum(box_mask * (gt_box[:, 1:2, :, :].unsqueeze(1) - pred_boxes[:, :, 1, :, :]) ** 2)
#     eps = 1e-6
#     loss_w = weight_coord * torch.sum(
#         box_mask * (
#             torch.sqrt(gt_box[:, 2:3, :, :].unsqueeze(1).clamp(min=eps)) -
#             torch.sqrt(pred_boxes[:, :, 2, :, :].clamp(min=eps))
#         ) ** 2
#     )
#     loss_h = weight_coord * torch.sum(
#         box_mask * (
#             torch.sqrt(gt_box[:, 3:4, :, :].unsqueeze(1).clamp(min=eps)) -
#             torch.sqrt(pred_boxes[:, :, 3, :, :].clamp(min=eps))
#         ) ** 2
#     )

#     #loss_obj = torch.sum(box_mask * (box_confidence - pred_boxes[:, :, 4, :, :]) ** 2)
#     loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
#     #loss_noobj = weight_noobj  * torch.sum((1.0-box_mask) * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
#     loss_noobj = weight_noobj * torch.sum((1.0 - box_mask) * torch.pow(output[:, 4:5*num_boxes:5], 2.0))
#     #loss_noobj = weight_noobj * torch.sum((1.0 - box_mask) * (pred_boxes[:, :, 4, :, :]) ** 2)

#     # loss_cls = torch.sum(gt_mask.unsqueeze(1) * (pred_cls - 1.0) ** 2)  # since it's a one-class problem
#     loss_cls = torch.sum(gt_mask.unsqueeze(1) * (pred_cls - 1.0)**2 + (1 - gt_mask.unsqueeze(1)) * (pred_cls)**2)
#     ### ADD YOUR CODE HERE ###
#     # Use weight_coord and weight_noobj defined above

#     # print('lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' % (loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj, loss_cls))

#     # the totol loss
#     loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
#     return loss

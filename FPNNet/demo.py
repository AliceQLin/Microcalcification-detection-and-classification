import cv2
import numpy
from torchvision import transforms
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def compute_iou(box1, box2):
    # box [x1, y1, x2, y2]
    box1 = box1.to(torch.device("cpu"))
    box2 = box2.to(torch.device("cpu"))
    rec1 = (box1[1], box1[0], box1[3], box1[2])
    rec2 = (box2[1], box2[0], box2[3], box2[2])

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def visualize(model, data_loader, device):
    model.eval()
    classes = ['gt', '0', '1']
    colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)]

    _id1, _id2, _id3 = [], [], []
    pred_list1, pred_score_list1, img_label_list1 = [], [], [] # patch(bbox)
    pred_list2, pred_score_list2, img_label_list2 = [], [], [] # image
    pred_list3, pred_score_list3, img_label_list3 = [], [], [] # patient
    
    for count, (images, targets, imgs_idx) in enumerate(data_loader):
        gt_boxes = targets[0]["boxes"]
        gt_labels = targets[0]["labels"]
        PIL_img = transforms.ToPILImage()(images[0])
        gt_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)
        rs_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        out = model(images, targets)
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']
        
        print()
        print("图%d" % (count + 1))
        print(imgs_idx[0][:-4])
        
        gt_label_tmp = classes[gt_labels[0].item()]
        for idx in range(gt_boxes.shape[0]):
            x1, y1, x2, y2 = gt_boxes[idx][0], gt_boxes[idx][1], gt_boxes[idx][2], gt_boxes[idx][3]
            name = classes[gt_labels[idx].item()]
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), colors[gt_labels[idx].item()], thickness=2)
            cv2.putText(gt_img, text=name, org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[gt_labels[idx].item()])
        
        boxes = list(out[0]['boxes'])
        labels = list(out[0]['labels'])
        scores = list(out[0]['scores'])
        ori_1 = len(boxes)
        
        boxes_del = []
        pd_boxes_count = len(boxes)
        for idx in range(pd_boxes_count):
            for pd_idx in range(pd_boxes_count):
                if pd_idx <= idx:
                    continue
                iou = compute_iou(boxes[idx], boxes[pd_idx])
                if iou > 0.5 and int(classes[labels[idx]]) != int(classes[labels[pd_idx]]):
                    if int(classes[labels[idx]]) == 1:
                        boxes_del.append(pd_idx)
                    else:
                        boxes_del.append(idx)                       
        if len(boxes_del) != 0:            
            boxes_final = []
            labels_final = []
            scores_final = []            
            for i in range(len(boxes)):
                if i not in boxes_del:
                    boxes_final.append(boxes[i])
                    labels_final.append(labels[i])
                    scores_final.append(scores[i])                    
            boxes = boxes_final
            labels = labels_final
            scores = scores_final        
        assert(len(boxes)==len(scores)==len(labels))
        assert((ori_1-len(boxes))==len(boxes_del))        
        
        pred_label_tmp = 0
        pred_score_tmp = 0
        pred_bscore_tmp = 0
        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = classes[labels[idx]]
            if int(name) != 0 and float(scores[idx]) >= float(pred_score_tmp):
                pred_label_tmp = name
                pred_score_tmp = float(scores[idx])
            if int(name) == 0 and float(scores[idx]) >= float(pred_bscore_tmp):
                pred_bscore_tmp = float(scores[idx])
            cv2.rectangle(rs_img, (x1, y1), (x2, y2), colors[labels[idx]], thickness=2)
            cv2.putText(rs_img, text=name + " %.3f" % scores[idx], org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[labels[idx]])
        
        gt_boxes, gt_labels, pd_boxes, pd_labels = gt_boxes, gt_labels, boxes, labels
        gt_boxes_count = gt_boxes.shape[0]
        pd_boxes_count = len(pd_boxes)
        for idx in range(gt_boxes_count):
            all_ious = {}
            for pd_idx in range(pd_boxes_count):
                iou = compute_iou(gt_boxes[idx], pd_boxes[pd_idx])
                if iou > 0:
                    all_ious[pd_idx] = iou
    
            gt_class = classes[gt_labels[idx].item()]
    
            if len(all_ious) > 0:
                max_iou_idx = sorted(all_ious.items(), key=lambda x: x[1], reverse=True)[0][0]
                max_iou_pd_class = classes[pd_labels[max_iou_idx]]
                if int(max_iou_pd_class) == 0:
                    max_iou_pd_score = 1 - float(scores[max_iou_idx])
                else:
                    max_iou_pd_score = float(scores[max_iou_idx])
                                
                pred_list1.append(int(max_iou_pd_class))
                pred_score_list1.append(float(max_iou_pd_score))
                img_label_list1.append(int(gt_class))
                _id1.append(imgs_idx[0][:-4])
        
        if int(pred_label_tmp) == 0:
            if float(pred_bscore_tmp) == 0:
                pred_score = 0
            else:
                pred_score = 1 - pred_bscore_tmp
        else:
            pred_score = pred_score_tmp

        pred_list2.append(int(pred_label_tmp))
        pred_score_list2.append(float(pred_score))
        img_label_list2.append(int(gt_label_tmp)) 
        _id2.append(imgs_idx[0][:-4])
        
        imgs_name = imgs_idx[0][:-4]
        if imgs_name[-1] == "C":
            patient_flg = int(pred_label_tmp)
            patient_score = float(pred_score)
        else:
            if patient_flg == int(pred_label_tmp):
                if patient_flg == 0:
                    patient_score = min(float(pred_score), patient_score)
                else:
                    patient_score = max(float(pred_score), patient_score)
            else:
                if patient_flg == 0:
                    if float(pred_score) > 0.5:
                        patient_flg = 1
                        patient_score = float(pred_score)
                else:
                    if patient_score <= 0.5:
                        patient_flg = 0
                        patient_score = float(pred_score)
            
            pred_list3.append(int(patient_flg))
            pred_score_list3.append(float(patient_score))
            img_label_list3.append(int(gt_label_tmp))
            _id3.append(imgs_idx[0][:-4])
            print("patient:", int(patient_flg), int(gt_label_tmp))
        
        cv2.imwrite('results/%s_gt.png' % imgs_idx[0][:-4], gt_img)
        cv2.imwrite('results/%s_pred2.png' % imgs_idx[0][:-4], rs_img)    

    print("\n---------Lesion Level-------------------")
    y_true = img_label_list1
    y_pred = pred_list1
    classify_report = metrics.classification_report(y_true, y_pred, digits=4)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
     
    fpr, tpr, thresholds = roc_curve(img_label_list1, pred_score_list1);
    roc_auc = auc(fpr, tpr) 
    plt.subplots(figsize=(8,5));
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc);
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('1 – specificity', fontsize=20);
    plt.ylabel('Sensitivity', fontsize=20);
    plt.title('ROC Curve', fontsize=20);
    plt.legend(loc="lower right", fontsize=16);    
    plt.savefig("ROC_bbox.png", dpi=300, bbox_inches='tight') #解决图片不清晰，不完整的问题
    plt.show()
    
    submission = pd.DataFrame({"ID": _id1, "Pred": pred_list1, "Label": img_label_list1, "Score": pred_score_list1})
    submission.to_csv('detect_submission_bbox.csv', index=True, header=True)

    
    print("\n---------Image Level-------------------")
    y_true = img_label_list2
    y_pred = pred_list2
    classify_report = metrics.classification_report(y_true, y_pred, digits=4)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('overall_accuracy: {0:f}'.format(overall_accuracy))

    fpr, tpr, thresholds = roc_curve(img_label_list2, pred_score_list2);
    roc_auc = auc(fpr, tpr) 
    plt.subplots(figsize=(8,5));
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc);
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('1 – specificity', fontsize=20);
    plt.ylabel('Sensitivity', fontsize=20);
    plt.title('ROC Curve', fontsize=20);
    plt.legend(loc="lower right", fontsize=16);    
    plt.savefig("ROC_image.png", dpi=300, bbox_inches='tight') #解决图片不清晰，不完整的问题
    plt.show()

    submission = pd.DataFrame({"ID": _id2, "Pred": pred_list2, "Label": img_label_list2, "Score": pred_score_list2})
    submission.to_csv('detect_submission_image.csv', index=True, header=True)
    
    print("\n---------Breast Level-------------------")
    y_true = img_label_list3
    y_pred = pred_list3
    classify_report = metrics.classification_report(y_true, y_pred, digits=4)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    
    fpr, tpr, thresholds = roc_curve(img_label_list3, pred_score_list3);
    roc_auc = auc(fpr, tpr) 
    plt.subplots(figsize=(8,5));
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc);
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('1 – specificity', fontsize=20);
    plt.ylabel('Sensitivity', fontsize=20);
    plt.title('ROC Curve', fontsize=20);
    plt.legend(loc="lower right", fontsize=16);    
    plt.savefig("ROC_patient.png", dpi=300, bbox_inches='tight') #解决图片不清晰，不完整的问题
    plt.show()

    submission = pd.DataFrame({"ID": _id3, "Pred": pred_list3, "Label": img_label_list3, "Score": pred_score_list3})
    submission.to_csv('detect_submission_patient.csv', index=True, header=True)
    

if __name__ == '__main__':
    import os
    import torch
    from main import get_transform
    from dataset import NBIDataset
    from torch.utils.data import DataLoader, Subset
    import utils
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has 3 classes - 2 class (Benign/Malignant) + background
    num_classes = 3

    # use our dataset and defined transformations
    root = "./MC_dataset"
    dataset_test = NBIDataset(os.path.join(root, "test"), get_transform(train=False))
    
    # define training and validation data loaders
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    
    box_score_threshs = [0.3]
    box_nms_threshs = [0.3]
    
    for bst in box_score_threshs:
        for bnt in box_nms_threshs:
            rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=250,
                                                                              box_score_thresh=bst, box_nms_thresh=bnt)
    
            # get number of input features for the classifier
            in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            model = rcnn_model
            model.to(device)

            model_path = os.path.join("./models", 'model_epoch-50.pt')
            net_state_dict = torch.load(model_path)
            model.load_state_dict(net_state_dict)

            visualize(model, data_loader_test, device)

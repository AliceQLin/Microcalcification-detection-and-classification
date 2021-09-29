import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=250,
                                                                  box_score_thresh=0.3, box_nms_thresh=0.3)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 3

# get number of input features for the classifier
in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
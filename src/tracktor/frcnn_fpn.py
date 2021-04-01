from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes, box_detections_per_img=300)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img, object_ind):
        device = list(self.parameters())[0].device
        img = img.to(device)
        detections = self(img)[0]
        #print(detections['labels'])
        if object_ind == -1:
            return (
            detections['boxes'].detach(),
            detections['labels'].detach(),
            detections['scores'].detach()
        )
        
        ind = (detections['labels'] == object_ind).nonzero().reshape(-1)
        #print(detections['scores'][ind])
        return (
            detections['boxes'][ind].detach(),
            detections['labels'][ind].detach(),
            detections['scores'][ind].detach()
        )

    def predict_boxes(self, boxes, labels, object_ind):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        
        if object_ind == -1:
            pred_boxes = torch.stack([pred_boxes[i, labels[i], :] for i in range(len(labels))], 0).detach()
            pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
            pred_scores = torch.stack([pred_scores[i, labels[i]] for i in range(len(labels))], 0).detach()
            return pred_boxes, pred_scores
        
        pred_boxes = pred_boxes[:, object_ind,:].detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, object_ind].detach()
        return pred_boxes, pred_scores

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])

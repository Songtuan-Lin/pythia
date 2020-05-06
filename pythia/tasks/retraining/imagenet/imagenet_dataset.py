import torch
import torch.nn as nn
import os
import json
import numpy as np
import cv2
from PIL import Image

from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.utils.general import get_pythia_root
from pythia.utils.text_utils import VocabFromText, tokenize

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.layers import nms


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

maskrcnn_checkpoint = os.path.join(get_pythia_root(), '../data', 'model_data/detectron_model.pth')

cfg.merge_from_file(os.path.join(get_pythia_root(), '../data', 'model_data/detectron_model.yaml'))
cfg.freeze()

class VQAMaskRCNNBenchmark(nn.Module):
    def __init__(self):
        super(VQAMaskRCNNBenchmark, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.model = build_detection_model(cfg)

        model_state_dict = torch.load(maskrcnn_checkpoint)
        load_state_dict(self.model, model_state_dict.pop("model"))

        # make sure maskrcnn_benchmark is in eval mode
        self.model.eval()

    def _features_extraction(self, output,
                                 im_scales,
                                 feature_name='fc6',
                                 conf_thresh=0.5):
        batch_size = len(output[0]["proposals"])
        # list[num_of_boxes_per_image]
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        # list[Tensor: (n_boxes_per_image, num_classes)]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        # list[Tensor: (n_boxes_per_image, 2048)]
        features = output[0][feature_name].split(n_boxes_per_image)
        # list[Tensor: (num_features_selected_per_image, 2048)]
        # list contain selected features per image
        features_list = []

        for i in range(batch_size):
            # reshape the bounding box to original size/coordinate
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            # Tensor: (n_boxes_per_image, num_classes)
            scores = score_list[i]
            # Tensor: (n_boxes_per_image, )
            # max_conf record the  heightest probs of the class
            # associate with each bounding box. If the heightest prob
            # of a box (say i) is smaller than threshold (conf_thresh), 
            # this box will not be select and max_conf[i] will be set
            # to 0 
            max_conf = torch.zeros((scores.shape[0])).to(device)

            for cls_ind in range(1, scores.shape[1]):
                # Tensor: (n_boxes_per_image, 1)
                # score for a specified class
                cls_scores = scores[:, cls_ind]
                # index of boxes that will be keep
                keep = nms(dets, cls_scores, conf_thresh)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])
            
            # select the top 100 boxes which contain an onject with
            # probability greater than conf_thresh(usually 0.5)
            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            features_per_image = features[i][keep_boxes]
            features_list.append(features_per_image)
        
        return features_list

    def forward(self, images, image_scales):
        images = to_image_list(images, size_divisible=32)
        images = images.to(device)
        # the returned features of maskrcnn_benchmark is the result of
        # roi pooling without average pooling
        output = self.model(images)
        features = self._features_extraction(output, image_scales)

        return features


class ImageNetDataset(BaseDataset):
    def __init__(self, dataset_type, config):
        super().__init__('imagenet', dataset_type, config)
        self.feature_extractor = VQAMaskRCNNBenchmark()
        self.feature_extractor.to(device)
        self.config = config
        # directly to store annotations(captions)
        self.annotation_dir = os.path.join(get_pythia_root(), config.data_root_dir, config.annotation_dir)
        # directory to store images
        self.image_dir = os.path.join(get_pythia_root(), config.data_root_dir, config.image_dir)
        self.annotations = []
        for annotation_file in os.listdir(self.annotation_dir):
            with open(os.path.join(self.annotation_dir, annotation_file)) as f:
                annotation = json.load(f)
                for item in annotation.items():
                    # each item in annotations is a (image_id, caption) tuple
                    self.annotations.append(item)
        self.init_processors()

    def _image_transform(self, image_path):
        '''
        Read an image and apply necessary transform

        Args:
            image_path (str): path to the image

        Returns:
            Tensor: image data
            int: scale used to resize image
        '''
        img = Image.open(image_path)
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def load_item(self, idx):
        sample = Sample()
        image_id = self.annotations[idx][0]
        image_folder = image_id.split('_')[0]
        caption = self.annotations[idx][1]
        tokens = tokenize(caption)
        tokens = ['<s>'] + tokens + ['</s>']
        # use text_processor to process caption
        # pad sequence, convert token to indices and add SOS, EOS token
        # text_processor already contains a pre-processor to tokenize caption
        caption_p = self.text_processor({'tokens': tokens})
        sample.text = caption_p['text']
        sample.caption_len = torch.tensor(len(tokens), dtype=torch.int)
        # sample.target = caption_p['text']
        sample.answers = torch.stack([caption_p['text']])
        # generate image features
        image_path = os.path.join(self.image_dir, image_folder, image_id)
        image, image_scale = self._image_transform(image_path)
        image_features = self.feature_extractor([image], [image_scale])
        image_features = image_features[0]
        sample.image_feature_0 = image_features
        return sample

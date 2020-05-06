import boto3
import os
import json
# import imagenet_dataset

from pythia.common.registry import registry
from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
from pythia.utils.general import download_file, get_pythia_root
from .imagenet_dataset import ImageNetDataset

@registry.register_builder("imagenet")
class ImageNetBuilder(BaseDatasetBuilder):
    def __init__(self):
        '''
        Pythia dataset builder, will be called in BaseTask
        '''
        super().__init__('imagenet')
        self.writer = registry.get("writer")
        
    def _load(self, dataset_type, config, *args, **kwargs):
        if dataset_type == 'train':
            self.dataset = ImageNetDataset(dataset_type, config)
        else:
            # use coco dataset to inference
            # therefore, in configuration file,
            # we should embedd configuration of coco
            # into it
            attributes = config['coco']
            coco_builder = registry.get_builder_class('coco')
            coco_builder_instance = coco_builder()
            coco_builder_instance.build(dataset_type, attributes)
            self.dataset = coco_builder_instance.load(dataset_type, attributes)
        return self.dataset
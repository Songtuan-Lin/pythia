import os
import argparse
import numpy as np

from pythia.utils.text_utils import tokenize

class AnnotationReader:
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir
        annotations = self.load_annotations()
        targets = self.build(annotations)
        self.targets = np.asarray(targets)
    
    def load_annotations(self):
        annotations = []
        for annotation_file in os.listdir(self.annotation_dir):
            with open(os.path.join(annotation_dir, annotation_file)) as f:
                annotation = json.load(f)
                for item in annotation.items():
                    annotations.append(item)
        return annotations

    def build(self, annotations):
        targets = []
        for annotation in annotations:
            image_id = annotation[0]
            image_name = image_id
            caption_str = annotation[1]
            caption_tokenes = tokenize(caption_str)
            caption_tokenes = ['<s>'] + caption_tokenes + ['</s>']
            reference_tokens = [caption_tokenes]
            feature_path = image_id + '.npy'
            target = {
                'image_id': image_id,
                'image_name': image_name,
                'caption_str': caption_str,
                'caption_tokenes': caption_tokenes,
                'reference_tokens': reference_tokens,
                'feature_path': feature_path
            }
            targets.append(target)
        return targets

    def save(self, output_dir):
        file_name = 'imdb_imagenet.npy'
        np.save(os.path.join(output_dir, file_name), self.targets)


parser = argparse.ArgumentParser()
parser.add_argument("--annotation_dir", type=str, help="annotation directory")
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    reader = AnnotationReader(args.annotation_dir)
    reader.save(args.output_dir)
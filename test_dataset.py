from pythia.utils.configuration import Configuration
from pythia.common.registry import registry
from pythia.tasks.retraining.imagenet.imagenet_builder import ImageNetBuilder

configuration = Configuration('pythia/common/defaults/configs/tasks/retraining/imagenet.yml')
configuration.freeze()
config = configuration.get_config()
registry.register("config", config)
registry.register("configuration", configuration)
dataset_config = config.task_attributes.retraining.dataset_attributes.imagenet
    
imagenet_builder = ImageNetBuilder()
dataset_train = imagenet_builder._load('train', dataset_config)
dataset_val = imagenet_builder._load('val', dataset_config)

print(dataset_train.load_item(6))
print('************************')
print(dataset_val.load_item(6))
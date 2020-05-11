import glob
import importlib
import os


from pythia.utils.configuration import Configuration
from pythia.common.registry import registry
from pythia.tasks.captioning.coco.builder import COCOBuilder


def setup_imports():
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("pythia_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_pythia_path = os.environ.get("PYTHIA_PATH")

        if environment_pythia_path is not None:
            root_folder = environment_pythia_path

        root_folder = os.path.join(root_folder, "pythia")
        registry.register("pythia_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    tasks_folder = os.path.join(root_folder, "tasks")
    tasks_pattern = os.path.join(tasks_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.common.meter")

    files = glob.glob(tasks_pattern, recursive=True) + \
            glob.glob(model_pattern, recursive=True) + \
            glob.glob(trainer_pattern, recursive=True)

    for f in files:
        if f.endswith("task.py"):
            splits = f.split(os.sep)
            task_name = splits[-2]
            if task_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.tasks." + task_name + "." + module_name)
        elif f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.models." + module_name)
        elif f.find("trainer") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.trainers." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            task_name = splits[-3]
            dataset_name = splits[-2]
            if task_name == "tasks" or dataset_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "pythia.tasks." + task_name + "." + dataset_name + "." + module_name
            )

setup_imports()
configuration = Configuration('pythia/common/defaults/configs/tasks/captioning/imagenet.yml')
configuration.freeze()
config = configuration.get_config()
registry.register("config", config)
registry.register("configuration", configuration)
dataset_config = config.task_attributes.captioning.dataset_attributes.coco
    
builder = COCOBuilder()
dataset_train = builder._load('train', dataset_config)
dataset_train.init_processors()
dataset_val = builder._load('val', dataset_config)
dataset_val.init_processors()

print(dataset_train.load_item(6))
print('************************')
print(dataset_val.load_item(6))

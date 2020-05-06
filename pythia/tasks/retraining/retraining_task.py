from pythia.common.registry import registry
from pythia.tasks import BaseTask


@registry.register_task('retraining')
class RetrainingTask(BaseTask):
    def __init__(self):
        super(RetrainingTask, self).__init__('retraining')
    
    def _get_available_datasets(self):
        return ['imagenet']

    def _preprocess_item(self, item):
        return item
"""Starts a job from a saved global model instead of the initial one."""

import numpy as np
from nvflare.app_common.abstract.model import make_model_learnable

from persistor import DictModelPersistor


class CheckpointPersistor(DictModelPersistor):
    def __init__(self, mib: int, checkpoint: str):
        super().__init__(mib)
        self.checkpoint = checkpoint

    def load_model(self, fl_ctx):
        with np.load(self.checkpoint) as saved:
            return make_model_learnable(weights=dict(saved), meta_props={})

"""NPModelPersistor holds one array, the shared workload is a dict of them. Otherwise the same."""

import os

import numpy as np
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor

from common import initial_model

MODEL_FILE = "global_model.npz"


class DictModelPersistor(ModelPersistor):
    def __init__(self, mib: int):
        super().__init__()
        self.mib = mib

    def load_model(self, fl_ctx):
        return make_model_learnable(weights=initial_model(self.mib), meta_props={})

    def save_model(self, model_learnable, fl_ctx):
        root = fl_ctx.get_workspace().get_result_root(fl_ctx.get_job_id())
        os.makedirs(root, exist_ok=True)
        np.savez(os.path.join(root, MODEL_FILE), **model_learnable[ModelLearnableKey.WEIGHTS])

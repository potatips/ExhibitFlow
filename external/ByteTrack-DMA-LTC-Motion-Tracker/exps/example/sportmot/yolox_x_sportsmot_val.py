# SportsMOT validation-set eval config (revision): same as yolox_x_sportsmot.py but val.json.
import os
from exps.example.sportmot.yolox_x_sportsmot import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.val_ann = "val.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

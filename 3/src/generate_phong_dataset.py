from collections import namedtuple
from enum import Enum
from importlib.resources import path

import moderngl_window
import pandas as pd

from phong_window import PhongWindow

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../output_dataset/"
        ],
        PhongWindow
    )

    # def read_csv(self, path):
    #     pd.set_option("display.max_columns", None)
    #     pd.set_option("display.width", 1000)
    #     df = pd.read_csv(path)
    #     print(df)

if __name__ == '__main__':
    task = TaskType.PHONG
    moderngl_window.run_window_config(task.window_cls, args=task.window_args)
    # task.read_csv("../output_dataset/dataset.csv")

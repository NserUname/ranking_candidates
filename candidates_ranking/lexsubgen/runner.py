import logging
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, NoReturn, Dict, Any
import os
# import fire
from tqdm import tqdm
import pandas as pd
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen.utils.file import create_run_dir, import_submodules, dump_json

from lexsubgen.utils.params import (
    build_from_config_path, build_from_params,read_config
)
from lexsubgen.utils.register import ENTRY_DIR

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

# 支持评估、超参数搜索、数据增强三大任务
class Runner:
    @staticmethod
    def import_additional_modules(additional_modules):
        # Import additional modules
        logger.info("Importing additional modules...")
        if additional_modules is not None:
            if not isinstance(additional_modules, list):
                additional_modules = [additional_modules]
            for additional_module in additional_modules:
                import_submodules(additional_module)

    def __init__(self,config: Dict[str,Any], run_dir: str, force: bool = False, auto_create_subdir: bool = False):
        """
        Class that handles command line interaction with the LexSubGen framework.
        Different methods of this class are related to different scenarios of framework usage.
        E.g. evaluate method performs substitute generator evaluation on the datasets specified
        in the configuration.
        Args:
            run_dir: path to the directory where to store experiment data.
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
        """
        self.run_dir = Path(run_dir)
        if auto_create_subdir and not force:
            time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

            sg_name = Path(config['substitute_generator']).name
            dr_name = Path(config['dataset_reader']).name
            # 新的目录名：模型+数据集
            new_dir_name = f"{time_str}_{sg_name}_{dr_name}"

            self.run_dir = self.run_dir / new_dir_name
            # self.run_dir = self.run_dir / f"{time_str}"
        self.force = force
        # Create run directory and write into config
        # logger.info(f"Creating run directory {self.run_dir}...")
        create_run_dir(self.run_dir, force=self.force)
        dump_json(Path(self.run_dir) / "config.json", config)

    # 配置文件实例化对象——传入lexsub——读取数据，计算metric，写入；无返回值
    def evaluate(self,config: Dict[str, Any],experiment_name: Optional[str] = None,run_name: Optional[str] = None):
        """
        Evaluates task defined by configuration file.
        Args:
            config_path: path to a configuration file.
            config: configuration of a task.
            additional_modules: path to directories with modules that should be registered in global Registry.
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        # 动态加载对象
        # 从配置中获取路径并构建对象，第二个参数是传入的config，不用返回
        substgen_config, _ = build_from_config_path(config["substitute_generator"])
        dataset_config, _ = build_from_config_path(config["dataset_reader"])

        # 构建 LexSubEvaluation 实例，传入的是真正的对象了
        lexsub_eval = LexSubEvaluation(
            substitute_generator=substgen_config,
            dataset_reader=dataset_config,
            verbose=config["verbose"],
            k_list=config["k_list"],
            batch_size=config["batch_size"],
            save_instance_results=config["save_instance_results"],
            save_wordnet_relations=config["save_wordnet_relations"],
            save_target_rank=config["save_target_rank"],
        )
        # 执行评测逻辑
        dataset = lexsub_eval.dataset_reader.read_dataset() 

        metrics = lexsub_eval.get_metrics(dataset)
        
        lexsub_eval.dump_metrics(metrics, self.run_dir, log=True)




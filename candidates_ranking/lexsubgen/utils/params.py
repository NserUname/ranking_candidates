import importlib
import json
from typing import Optional
from collections.abc import MutableMapping
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import _jsonnet
from _jsonnet import evaluate_file
from lexsubgen.utils.register import logger
import os
# 从config配置文件中构建对象

class Params(MutableMapping):
    DEFAULT_VALUE = object

    def __init__(self, params: Dict):
        """
        Objects of this class represents parameters dictionary.
        Using this class one could build other objects with build_from_params function.
        You may consume parameters with pop method and at the end check that all parameters
        were read. More precisely its a wrapper around ordinary `dict` which supports some
        auxiliary functionality.

        Args:
            params: dictionary representing parameters.
        """
        self.__dict__.update(params)

    @property
    def dict(self) -> Dict:
        """
        Get underlying parameters dictionary.

        Returns:
            parameters dictionary
        """
        return self.__dict__

    def get(
        self, key: str, default_value: object = DEFAULT_VALUE, _type: Optional = None
    ):
        """
        Implements functionality of `dict.get(key)` method but also check for the the type of
        returned value. If it's a dict then it will convert it into Params object. Also it supports
        conversion of return value to the specified standard type.

        Args:
            key: identifier of the object to get from the Params
            default_value: default value to return if there is no object with specified key
            _type: type to which the return object should be converted

        Returns:
            object from Params
        """
        if default_value is self.DEFAULT_VALUE:
            value = self.__dict__.get(key)
        else:
            value = self.__dict__.get(key, default_value)
        value = self._convert_value(value)
        if _type is not None:
            value = self._convert_type(value, _type)
        return value

    def pop(
        self, key: str, default_value: object = DEFAULT_VALUE, _type: Optional = None
    ):
        """
        Performs functionality of `dict.pop` method but additionally converts dict return object
        into Params objects. Also return values could be converted to the specified type. If there is no
        object with specified key then default value will be return if it's given.
        The object with specified key will be removed from Params.

        Args:
            key: identifier of the object to get from the Params
            default_value: default value to return if there is no object with specified key
            _type: type to which the return object should be converted

        Returns:
            object from Params
        """
        if default_value is self.DEFAULT_VALUE:
            value = self.__dict__.pop(key)
        else:
            value = self.__dict__.pop(key, default_value)
        value = self._convert_value(value)
        if _type is not None:
            value = self._convert_type(value, _type)
        return value

    def _convert_value(self, value):
        """
        Checks the type of the value.
        If it's a dict then converts it to Params object.
        If it's a list then it goes through the list and subsequently performs
        conversion of each element.

        Args:
            value: object to process

        Returns:
            converted object
        """
        if isinstance(value, dict):
            return Params(value)
        elif isinstance(value, list):
            value = [self._convert_value(item) for item in value]
        return value

    @staticmethod
    def _convert_type(value, _type):
        """
        Converts object to the given type.

        Args:
            value: object to be converted
            _type: type to which the object should be converted

        Returns:
            converted object
        """
        if value is None or value == "None":
            return None
        if _type is bool:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value == "false":
                    return False
                if value == "true":
                    return True
                raise TypeError("To convert to bool value should be bool or str.")
        return _type(value)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self._convert_value(self.__dict__[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)



#______________________ jsonnet转为json
# 读取 Jsonnet或者json配置文件并返回为 【字典】
def read_config(config_path: str, verbose: bool = False):
    """
    Builds config object from configuration file
    Args:
        config_path: path to a configuration file in json or jsonnet format.
        verbose: Bool flag for verbosity.
    Returns:
        Built config object
    """
    logger.info("Loading configuration file...")
    config_path = Path(config_path) # 将一个字符串类型的路径 config_path 转换为 Path 对象。这样做的好处是可以使用 Path 对象的各种方法和属性来更方便地操作路径。
    os.chdir(config_path.parent)    # 切换到 配置文件的路径

    if config_path.suffix == ".jsonnet":
        config = json.loads(evaluate_file(str(config_path)))
    elif config_path.suffix == ".json":
        with open(config_path, "r") as fp:
            config = json.load(fp)
    else:
        raise ValueError(
            "Configuration files should be provided in json or jsonnet format"
        )
    logger.info(f"Loaded configuration: {config}")
    # print("read_config: 返回的是字典类型！")
    return config

# 传入配置文件 str 路径，返回实例化后的对象（模型+data）
def build_from_config_path(config_path: str) -> Tuple[Any, dict]:

    config = read_config(config_path)
    class_name = config.get("class_name", "")

    # 根据配置文件路径推断默认模块前缀
    config_path = Path(config_path).as_posix()  # 转换为 POSIX 路径格式
    if "subst_generators" in config_path:
        default_module = "lexsubgen.subst_generator"
    elif "dataset_readers" in config_path:
        default_module = "lexsubgen.datasets.lexsub"   
    else:
        default_module = "lexsubgen"  # 默认顶层模块

    # 自动补全 class_name
    if "." not in class_name:
        class_name = f"{default_module}.{class_name}"

    # 分割模块名和类名
    module_name, cls_name = class_name.rsplit(".", 1)
    # 提取所有参数（排除 class_name）
    params = {k: v for k, v in config.items() if k != "class_name"}
    # 动态导入类
    # print("导入的模块名：")
    # print(module_name)

    module = importlib.import_module(module_name)   # 动态导入模块
    cls = getattr(module, cls_name)
    # 实例化对象（传递所有参数）
    instance = cls(**params)  # 直接传递 params 而非 config.get("params", {})   会进入estimator.py
    return instance, config

# 在合适的地方实现一个新的函数，例如在subst_generator.py中_____-解决preprocessor和postprocessor
def build_from_dict_config(config_input):
    """
    修改后的函数，支持直接传入字典或配置文件路径
    """
    # 如果输入是字典，直接使用
    if isinstance(config_input, dict):
        config = config_input
    # 否则视为文件路径，读取配置
    else:
        config = read_config(config_input)
    # 提取类名
    class_name = config.get("class_name", "")
    # 如果输入是字典，尝试从字典中获取模块路径提示（如果存在）
    if isinstance(config_input, dict):
        # 从字典中获取模块提示（例如通过添加自定义字段）
        module_hint = config.get("module_hint", "")
    else:
        # 根据文件路径推断模块前缀
        config_path = Path(config_input).as_posix()  # 转换为POSIX路径
        if "subst_generators" in config_path:
            module_hint = "lexsubgen.subst_generator"
        elif "dataset_readers" in config_path:
            module_hint = "lexsubgen.datasets.lexsub"
        else:
            module_hint = "lexsubgen"
    # 自动补全类名
    if "." not in class_name:
        class_name = f"{module_hint}.{class_name}"
    # 分割模块名和类名
    module_name, cls_name = class_name.rsplit(".", 1)

    # 提取参数（排除class_name）
    params = {k: v for k, v in config.items() if k != "class_name"}

    # 动态导入类
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)

    # 实例化对象
    instance = cls(**params)
    return instance, config

def clsname2cls(clsname: str):
    import_path = "lexsubgen"
    if "." in clsname:
        module_path, clsname = clsname.rsplit(".", 1)
        import_path += "." + module_path
    # try:
    module = importlib.import_module(import_path)
    cls = getattr(module, clsname)
    # except Exception as e:
    #     raise ValueError(f"Failed to import '{clsname}'")
    return cls


def build_from_params(params: Params):
    """
    Builds object from parameters. Parameters must contain a 'class_name' field
    indicating where to import the class from.

    Args:
        params: parameters from that the object will be build.

    Returns:
        object that was build from parameters.
    """
    if isinstance(params, int):
        return params

    if "class_name" in params:
        cls_name = params.pop("class_name")
        cls = clsname2cls(cls_name)

        # init params acquisition
        kwargs = {}
        keys = list(params.keys())
        for key in keys:
            item_params = params.pop(key)
            if isinstance(item_params, Params):
                item = build_from_params(item_params)
            elif isinstance(item_params, list):
                item = [build_from_params(elem_params) for elem_params in item_params]
            else:
                item = item_params
            kwargs[key] = item
        return cls(**kwargs)
    return params


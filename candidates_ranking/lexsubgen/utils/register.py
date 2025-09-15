'''
方便地管理项目的路径和日志，同时利用函数缓存提高程序的运行效率。
'''
import logging
import os
from pathlib import Path
from joblib import Memory

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

ENTRY_DIR = Path(__file__).resolve().parent.parent
print(f'entry_dir:{ENTRY_DIR}')
CONFIGS_DIR = ENTRY_DIR / "configs"
LOGS_DIR = ENTRY_DIR / "logs"

CACHE_DIR = Path(os.environ["HOME"]) / ".cache" / "lexsubgen"
# 这里代码试图从 os.environ 中获取 HOME 环境变量的值，但在 Windows 系统中，os.environ 中没有 HOME 这个键，因此会抛出 KeyError: 'HOME' 异常。
# CACHE_DIR = Path(os.environ["USERPROFILE"]) / ".cache" / "lexsubgen"
# linux运行，需要改成：USERPROFILE
# file_path = PosixPath('/root/.cache/lexsubgen/datasets/source/data/semeval_all/gold'
DATASETS_DIR = ENTRY_DIR / "data"/"source"/"data"   # 数据集的文件夹位置
MEMORY_CACHE_PATH = CACHE_DIR / "function_cache"
memory = Memory(str(MEMORY_CACHE_PATH), verbose=0)

# 方便下载huggingface模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


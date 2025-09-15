'''
下载数据集+基本的数据处理操作
'''
import logging
import os
import shutil
import unicodedata
from pathlib import Path
from typing import List
import wget
from lexsubgen.utils.file import extract_archive

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)
# 数据集地址：coinco semeval
LEXSUB_DATASETS_URL = "https://github.com/stephenroller/naacl2016/archive/master.zip" # 一致性

# 从给定的 URL 下载数据集，并将其保存到指定的目录中。
def download_dataset(url: str, dataset_path: str):
    """
    Method for downloading datasets from a given URL link.
    After download datasets will be saved in the dataset_path directory.
    Args:
        url: URL link to datasets.
        dataset_path: Directory path to save the downloaded datasets.
    Returns:
    """
    os.makedirs(dataset_path, exist_ok=True)
    logger.info(f"Downloading file from '{url}'...")
    filename = wget.download(url, out=str(dataset_path))
    logger.info(f"File {filename} is downloaded to '{dataset_path}'.")
    filename = Path(filename)

    # Extract archive if needed
    extract_archive(arch_path=filename, dest=dataset_path)

    # Delete archive
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)

# 去除字符串中的重音符号。
# 它会将字符串进行 NFD（Normalization Form D）规范化，这种规范化会将字符分解为基本字符和组合字符（例如重音符号）。
# 然后，它会过滤掉所有分类为 Mn（Mark, Nonspacing）的字符，也就是那些重音符号，最后将剩余的字符组合成一个新的字符串。
def strip_accents(s: str) -> str:
    """
    Remove accents from given string:
    Example: strip_accents("Málaga") -> Malaga
    Args:
        s: str - string to process
    Returns:
        string without accents
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

# 输入的字符串按指定的分隔符进行分割。
def split_line(line: str, sep: str = " ") -> List[str]:
    """
    Method for splitting line by given separator 'sep'.

    Args:
        line: Input line to split.
        sep: Separator char.
    Returns:
        line: List of parts of the input line.
    """
    line = [part.strip() for part in line.split(sep=sep)]
    return line

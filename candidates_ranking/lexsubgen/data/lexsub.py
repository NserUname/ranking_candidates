'''
用于读取词汇替换（Lexical Substitution）数据集的模块。
'''
import logging
from pathlib import Path
from typing import List, Union, Optional
import pandas as pd
from overrides import overrides
from lexsubgen.data.utils import split_line
from lexsubgen.utils.register import DATASETS_DIR
from .base_reader import DatasetReader
import ast
# from allennlp.data.dataset_readers.dataset_reader import DatasetReader


logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

# Links to basic supported lexical substitution data sets.
LEXSUB_DATASET_DRIVE_URLS = {
    # 斯图加特大学的官方网站下载CoInCo数据集:https://www.ims.uni-stuttgart.de/en/research/resources/corpora/coinco/
    # "coinco": "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/coinco/coinco.xml.gz",
    # 没有sentence gold之类的文件，和如下的不同，暂时不考虑
    # "semeval_all": "https://docs.google.com/uc?export=download&id=1TG-B09n2K5oRd_tJzMlBNhe0Jr_89s5c", # 2003句
    "semeval_all":"https://gitee.com/zldn/lexsub/tree/master/data/semeval_all",
    "coinco":"https://gitee.com/zldn/lexsub/tree/master/data/coinco",
    "swords": "https://github.com/……s/swords",
    # "semeval_test": "https://docs.google.com/uc?export=download&id=1StQwn2d1eYy3phHfWqAyRYE7CTLsO2pg",
    # "semeval_trial": "https://docs.google.com/uc?export=download&id=1SiPovrnD_EMrdhkyII3Vkw-jinUZZBqn",
    # "twsi2": "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2768/TWSI2_complete.zip?sequence=3&isAllowed=y",
    # twsi2的原始数据有点脏数据
}

# List of supported data sets.
LEXSUB_DATASETS = ("semeval_all", "coinco", "twsi2","swords")


# 读取semeval\coinco\twsi2……
class LexSubDatasetReader(DatasetReader):
    DATA_COLUMNS = [
        "context",
        "candidates",
        "target_position",
        "target_lemma",
        "pos_tag",
        "gold_subst",
        "gold_subst_weights",
    ]

    def __init__(
        self,
        dataset_name: str,
        data_root_path: Union[str, Path] = DATASETS_DIR,
        url: Optional[str] = None,
        with_pos_tag: bool = True,
    ):
        """
        Reader for Lexical Substitution datasets.
        Args:
            dataset_name: Alias for datasets naming.
            data_root_path: Path for all available datasets. Datasets will be downloaded to this directory.
            url: Link for downloading datasets.
            with_pos_tag: Bool flag. If True, then the reader expects the presence of POS-tags in the datasets.
        """
        if url is None and dataset_name in LEXSUB_DATASETS:
            # datasets from list of supported lexsub data sets
            url = LEXSUB_DATASET_DRIVE_URLS[dataset_name]
        super(LexSubDatasetReader, self).__init__(      # 数据集文件地址确定
            dataset_name=dataset_name, data_root_path=data_root_path, url=url
        )
        self.with_pos_tag = with_pos_tag

    @overrides
    def read_dataset(self) -> pd.DataFrame:
        """
        Lexical Substitution datasets consists of 3 different files:
            1. sentences - file with contexts, target word positions and POS-tags.
            2. golds - file with gold substitutes and annotators info.
            3. candidates - file with candidates for Candidate Ranking task.
        Returns:
            `pandas.DataFrame`: dataframe that contains following columns:
                'context', 'gold_subst', 'gold_subst_weights', 'target', 'target_position',
                'pos_tag', 'candidates'
        """
        golds_data = self._preprocess_gold_part(
            self.read_file(self.dataset_path / "gold")  # 少了7条数据
        )
        sentences_data = self._preprocess_sentence_part(
            self.read_file(self.dataset_path / "sentences")
        )
        candidates_data = self._preprocess_candidate_part(
            self.read_file(self.dataset_path / "candidates")
        )
        # Reading mapping from target to candidates
        lemma_to_candidates = {}
        for lemma, *candidates in candidates_data:
            lemma_to_candidates[lemma] = list(sorted(set(candidates)))

        # Reading golds
        golds_map = {}
        for datum in golds_data:
            gold_id = datum[1]
            assert gold_id not in golds_map, "Duplicated gold id occurred!"
            substitutes = [pair[0] for pair in datum[2:] if pair]
            gold_weights = [float(pair[1]) for pair in datum[2:] if pair]
            golds_map[gold_id] = {
                "gold_subst": substitutes,
                "gold_subst_weights": gold_weights,
            }

        # Reading context and creating datasets
        dataset = {column: [] for column in self.DATA_COLUMNS}
        for datum in sentences_data:
            context_id = datum[1]
            if context_id not in golds_map:
                logger.warning(f"Missing golds for context with id {context_id}")
                continue

            if self.with_pos_tag:
                target, pos_tag = datum[0].split(".", maxsplit=1)
                dataset["target_lemma"].append(target)
                dataset["pos_tag"].append(pos_tag)
                cands = lemma_to_candidates[target + "." + pos_tag.split(".")[0]]
                dataset["candidates"].append(cands)
            else:
                target = datum[0]
                dataset["target_lemma"].append(target)
                dataset["pos_tag"].append(None)
                dataset["candidates"].append(lemma_to_candidates[target])
            dataset["target_position"].append(int(datum[2]))
            dataset["context"].append(datum[3].split())
            gold_data = golds_map[context_id]
            dataset["gold_subst"].append(gold_data["gold_subst"])
            dataset["gold_subst_weights"].append(gold_data["gold_subst_weights"])
            assert dataset["target_position"][-1] <= len(dataset["context"][-1]), \
                f"Wrong target position ({dataset['target_position']} in context with id {context_id})"

        return pd.DataFrame(data=dataset)

    @staticmethod
    def _preprocess_sentence_part(sentences: List[str]):
        """
        Method for processing raw lines from file with sentences.

        Args:
            sentences: List of raw lines.
        Returns:
            sentences: List of processed sentences.
        """
        for idx in range(len(sentences)):
            sentence_info = split_line(sentences[idx], sep="\t")
            sentences[idx] = sentence_info
        return sentences

    @staticmethod
    def _preprocess_candidate_part(candidates):
        """
        Method for processing raw lines from file with candidates.

        Args:
            candidates: List of raw lines.
        Returns:
            candidates: List of processed candidates.
        """
        for idx in range(len(candidates)):
            candidates_info = split_line(candidates[idx], sep="::")
            candidates[idx] = [candidates_info[0].strip()]
            candidates[idx] += candidates_info[1].split(";")
            for jdx in range(1, len(candidates[idx])):
                candidates[idx][jdx] = candidates[idx][jdx].strip()
        return candidates

    @staticmethod
    def _preprocess_gold_part(golds):
        """
        Method for processing raw lines from file with golds.

        Args:
            golds: List of raw lines.
        Returns:
            golds: List of processed golds.
        """
        for idx in range(len(golds)):
            gold_info = split_line(golds[idx], sep="::")
            golds[idx] = gold_info[0].rsplit(maxsplit=1)
            golds[idx].extend([
                tuple(subst.strip().rsplit(maxsplit=1))
                for subst in gold_info[1].split(";")
                if subst
            ])
        return golds


# 读取swords数据集的Reader:注意是xlsx格式
# swords数据集的格式和semeval/coinco/twsi2不一样
class LexSubSwordsReader(DatasetReader):
    DATA_COLUMNS = [
        "context",
        "target_position",
        "candidates",
        "target_lemma",
        "pos_tag",
        "gold_subst",
        "gold_subst_weights",
    ]

    def __init__(
        self,
        dataset_name: str,
        data_root_path: Union[str, Path]=DATASETS_DIR,
        url: Optional[str] = None,
        with_pos_tag: bool = True,
    ):
        """
        Reader for Lexical Substitution datasets in .xlsx format.
        """
        self.dataset_name = dataset_name
        self.data_root_path = Path(data_root_path)
        self.url = url or LEXSUB_DATASET_DRIVE_URLS.get(dataset_name, None)
        
        super(LexSubSwordsReader, self).__init__(      # 数据集文件地址确定
            dataset_name=dataset_name, data_root_path=data_root_path, url=url
        )
        self.with_pos_tag = with_pos_tag

    @overrides
    def read_dataset(self) -> pd.DataFrame:
        """
        Read dataset from local Excel (.xlsx) file.
        Expects:
            - swords.xlsx in data_root_path/dataset_name/
        Returns:
            pandas.DataFrame with columns:
            context, target_position, pos_tag, gold_subst, gold_subst_weights
        """
        dataset_path = self.data_root_path / self.dataset_name
        xlsx_path = dataset_path / f"{self.dataset_name}.xlsx"

        # 读取 Excel 文件
        df = pd.read_excel(xlsx_path)

        # 确保包含所需列
        missing_cols = [col for col in self.DATA_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Excel 文件缺少列: {missing_cols}")

        # 只保留需要的列，自动转换 context 为分词列表
        df = df[self.DATA_COLUMNS].copy()
        
        def parse_context(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)  # 如果是 '["a","b"]' 这种
                except (SyntaxError, ValueError):
                    return x.split()            # 如果是 "a b c"
            return x

        df["context"] = df["context"].apply(parse_context)
        
        # 把字符串格式的列表转成真正的列表
        
        df["gold_subst"] = df["gold_subst"].apply(ast.literal_eval)
        df["gold_subst_weights"] = df["gold_subst_weights"].apply(ast.literal_eval)
        df["candidates"] = df["candidates"].apply(ast.literal_eval)
        
        return df

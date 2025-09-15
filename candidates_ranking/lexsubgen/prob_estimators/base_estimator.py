'''
抽象基类，用于概率估计器。该类包含了一些基本的方法和属性，用于处理日志记录和计算词汇概率分布。

'''

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from scipy.special import softmax


class BaseProbEstimator:
    def __init__(self, verbose: bool = False,weights:str=None,
                 stratagy_input_embedding:str=None,generation_otherPosition_ways:str=None,generation_otherPosition_ways_mode:str=None):
        """
        Abstract class that defines basic methods for probability estimators.

        Args:
            verbose: whether to print misc information
            verbose:冗长的、啰嗦的，控制是否打印详细信息
        """
        self.verbose = verbose
        self.weights=weights
        self.stratagy_input_embedding=stratagy_input_embedding
        self.generation_otherPosition_ways=generation_otherPosition_ways
        self.generation_otherPosition_ways_mode=generation_otherPosition_ways_mode

        self.logger = logging.getLogger(Path(__file__).name)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        self.output_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.output_handler.setFormatter(formatter)
        self.logger.addHandler(self.output_handler)
    # 计算每个目标词在词列表中的概率，以批处理方式处理所有输入数据。
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.

        Examples:
            >>> token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
            >>> target_ids_list = [1,2]
            >>> self.get_log_probs(tokens_lists, target_ids)
            # This means that we want to get probability distribution for words "world" and "stackoverflow".
        """
        raise NotImplementedError()
    # 计算给定实例的词汇概率分布。
    def get_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes probability distribution over vocabulary for a given instances.

        Args:
            tokens_lists: list of contexts.
            target_ids: list of target word ids.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            Probability distribution over vocabulary and the relative vocabulary.
        """
        logits, word2id = self.get_log_probs(tokens_lists, target_ids,target_pos_tag)
        probs = softmax(logits, axis=-1)
        return probs, word2id
    
    def get_model_score(self,sentences: List[List[str]] = None,pred_substitutes: List[str]=None,target_ids: List[List[int]]=None):
        raise NotImplementedError()
    
    def remove_special_tokens(self,pred_substitutes: List[List[str]]):
        raise NotImplementedError()
    
    def get_ordered_synonyms(self,original_word:str,synonyms_from_wordnet: List[str]):
        raise NotImplementedError()

    def getTargetPositionConcatScore(self,token_lists: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[List[int]]=None):
        raise NotImplementedError()
    
    def getOtherPositionConcatScoreByIg(self,tokens_lists: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[List[int]]=None):
        raise NotImplementedError()

    
    def getOtherPositionConcatScoreByAtten(self,tokens_lists: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[List[int]]=None):
        raise NotImplementedError()
    
    def getOtherPositionConcatScoreByOne(self,tokens_lists: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[List[int]]=None):
        raise NotImplementedError()

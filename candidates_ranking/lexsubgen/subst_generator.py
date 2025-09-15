'''
基类——模型继承自这个类
'''

from itertools import groupby
from typing import Iterable, Optional, List, Tuple, Dict, Union
import numpy as np
from scipy.special import softmax
from lexsubgen.post_processors import PostProcessor
from lexsubgen.pre_processors import Preprocessor
from lexsubgen.prob_estimators import BaseProbEstimator
from lexsubgen.utils.params import build_from_config_path,build_from_dict_config

# 这里的probs需要留意,每一行中概率最高的 k 个元素的索引，时间复杂度O(n)
def top_k_strategy(probs: np.ndarray, k: int,id2word: dict) -> List[List[int]]:
    """
    Function that implements top-k strategy, i.e. chooses k substitutes with highest probabilities.
    Args:
        probs: probability distribution
        k: number of top substitutes to take
    Returns:
        list of chosen indexes
    """
    parted = np.argpartition(probs, kth=range(-k, 0), axis=-1)  # 分区，最大的排在右边
    sorted_ids = parted[:, -k:][:, ::-1]        # 排序,从右向左，步长为1，即，选出的topk是依照概率从高到低的——肯定有概率选到原始词
    
    sorted_probs = np.take_along_axis(probs, sorted_ids, axis=-1)
    # # 组合 id 和概率返回
    result = []
    for ids, scores in zip(sorted_ids, sorted_probs):
        # 构造每个候选词对应的字典
        candidate_list = {id2word[id_]:float(score) for id_, score in zip(ids, scores)}
        result.append(candidate_list)

    return sorted_ids.tolist(),result  

#  top-p 策略在生成文本时能够更好地平衡多样性和连贯性。
def top_p_strategy(_probs: np.ndarray, p: float) -> List[List[int]]:
    """
    Function that implement top-p strategy. Takes as much substitutes as to fulfill probability threshold.
    Args:
        _probs: probability distribution
        p: probability threshold
    Returns:
        list of chosen indexes
    """
    bs, vs = _probs.shape
    sorted_ids = np.argsort(_probs, axis=-1)[:, ::-1]
    sorted_probs = _probs[[[i] * vs for i in range(bs)], sorted_ids]
    cumsum_probs = np.cumsum(sorted_probs, axis=-1) - sorted_probs
    selected_ids = []
    for idx, group in groupby(np.argwhere(cumsum_probs < p).tolist(), lambda x: x[0]):
        selected_ids.append([pair[1] for pair in group])
    return selected_ids


class SubstituteGenerator:
    def __init__(
        self,
        prob_estimator: BaseProbEstimator,
        pre_processing: Optional[Iterable[Preprocessor]] = None,
        post_processing: Optional[Iterable[PostProcessor]] = None,
        substitute_handler=None,                                # 预测候选词的时候有用，好像并没有用到（涉及到带标志的）
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """
        Class that handles generation of a substitutes. This process is splitted onto
        pre-processing, probability estimation and post-processing.

        Args:
            prob_estimator: probability estimator to be used for probability
                distribution acquisition over possible substitutes
            pre_processing: list of object that do pre-processing of original contexts
            post_processing: list of objects that do post-processing with the acquired probability distribution
            substitute_handler: processes predicted substitutes, it can lemmatize them or exclude target word
                处理预测的替代词，可以进行词形还原或排除目标词。
            top_k: how many substitutes to grab according to top-k strategy
            top_p: probability threshold in top-p strategy
        """
        assert (
            top_k is None or top_p is None
        ), "You shouldn't provide values for top_k and top_p methods simultaneously."
        self.prob_estimator = prob_estimator
        self.pre_processing = pre_processing or []
        self.post_processing = post_processing or []
        self.substitute_handler = substitute_handler
        if top_k is not None and top_k <= 0:
            raise ValueError("k in top-k strategy must be non-negative!")
        self.top_k = top_k
        if top_p is not None and 0.0 <= top_p <= 1.0:
            raise ValueError("p in top-p strategy should be a valid probability value!")
        self.top_p = top_p

        # 如果 prob_estimator 是字典，则动态构建实例
        if isinstance(prob_estimator, dict):
            self.prob_estimator, _ = build_from_dict_config(prob_estimator)
        else:
            self.prob_estimator = prob_estimator  # 直接使用实例

    @classmethod
    def from_config(cls, config_path):
        """
        Create substitute generator from configuration.

        Args:
            config_path: path to configuration file.

        Returns:
            object of the SubstituteGenerator class.
        """
        subst_generator, _ = build_from_config_path(config_path)
        return subst_generator

    def get_log_probs(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos_tags: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.
        Args:
            sentences: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
            target_pos_tags: list of target pos tags    (词性)
                E.g.:
                sentences = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                target_pos_tags = ['n', 'n']
                This means that we want to get probability distribution for words "world" and "stackoverflow" and
                the targets are nouns
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """
        for preprocessor in self.pre_processing:
            # print(f"preprocessor是:{type(preprocessor)}")
            # 告知preprocessor是字典而非对象,转一下__________关键！！！！！！！！
            preprocessor,_ = build_from_dict_config(preprocessor)
            sentences, target_ids = preprocessor.transform(sentences, target_ids)

        
        log_probs, word2id = self.prob_estimator.get_log_probs(sentences, target_ids,target_pos_tags)
        for postprocessor in self.post_processing:
            target_words = [
                sentence[target_id] for sentence, target_id in zip(sentences, target_ids)
            ]
            
            postprocessor,_ = build_from_dict_config(postprocessor)
            log_probs, word2id = postprocessor.transform(
                log_probs,
                word2id,
                target_words=target_words,
                target_pos=target_pos_tags,         
            )
        return log_probs, word2id

    def get_probs(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes probability distribution over vocabulary for a given instances.

        Args:
            sentences: list of contexts.
            target_ids: list of target word ids.
            target_pos: target word pos tags
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            Probability distribution over vocabulary and the relative vocabulary.
        """
        log_probs, word2id = self.get_log_probs(sentences, target_ids, target_pos)
        probs = softmax(log_probs, axis=-1) # -1表示最后一个维度进行
        return probs, word2id


    # 根据概率分布选择替代词。它支持 top-k 或 top-p 策略来决定哪些词作为替代词。
    def substitutes_from_probs(
        self,
        probs: np.ndarray,      # bert的话就是哪个target位置处的softmax概率，(batch_size,vocabsize)
        word2id: Dict,
        sentences: List[List[str]] = None,
        target_ids: List[int] = None,
        target_pos: Optional[List[str]] = None,
        target_lemmas: Optional[List[str]] = None
    ):
        id2word = {idx: word for word, idx in word2id.items()}
        # 有出现原词的情况，不在这儿处理

        if self.top_k is not None:
            selected_ids,selected_ids_substitutes = top_k_strategy(probs, self.top_k,id2word)
        elif self.top_p is not None:
            selected_ids = top_p_strategy(probs, self.top_p)
        else:
            selected_ids = np.argsort(probs)[::-1]  # 返回数组元素排序后的索引。
        substitutes = [[id2word[idx] for idx in ids] for ids in selected_ids]
        

        return substitutes,selected_ids_substitutes

    @staticmethod
    def candidates_from_probs(
        probs: np.ndarray,
        word2id: Dict[str, int],
        candidates: List[List[str]],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Ranking candidates using probability distributions over vocabulary
        Args:
            probs: probability distributions over vocabulary
            word2id: mapping from word to its index in the vocabulary
            candidates: lists of candidates to be ranked
        Returns:
            ranked_candidates_in_vocab: Ranked @candidates that are in vocabulary
            ranked_candidates: Ranked @candidates
        """
        ranked_candidates_in_vocab, ranked_candidates = [], []
        for i in range(probs.shape[0]):
            # 首先计算在模型词典中的候选词排名，排除不在的
            candidates_in_vocab = [w for w in candidates[i] if w in word2id]
            candidate_scores = np.array([
                probs[i, word2id[cand]] for cand in candidates_in_vocab
            ])
            candidate2rank = {
                word: (candidate_scores > score).sum() + 1
                for word, score in zip(candidates_in_vocab, candidate_scores)
            }
            candidate2rank = sorted(candidate2rank.items(), key=lambda x: x[1])

            ranked_in_vocab_local = [word for word, _ in candidate2rank]

            ranked_local = ranked_in_vocab_local.copy()
            for word in candidates[i]:
                if word not in word2id:
                    ranked_local.append(word)

            ranked_candidates_in_vocab.append(ranked_in_vocab_local)
            ranked_candidates.append(ranked_local)

        return ranked_candidates_in_vocab, ranked_candidates
    
    @staticmethod
    def candidates_from_final_scores(
        candidates: List[List[str]],
        pred_substitutes_and_final_scores):
        
        ranked_candidates = []
        for cand_row, score_dict in zip(candidates, pred_substitutes_and_final_scores):
            # 只保留在score_dict中的候选词，并按分数降序排列
            ranked_row = sorted(
                [w for w in cand_row if w in score_dict],
                key=lambda w: score_dict[w],
                reverse=True
            )
            ranked_candidates.append(ranked_row)

        return ranked_candidates


    def generate_substitutes(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos: Optional[List[str]] = None,
        return_probs: bool = False,
        target_lemmas: Optional[List[str]] = None
    ) -> Union[Tuple[List[List[str]], Dict, np.ndarray], Tuple[List[List[str]], Dict]]:
        """
        Generates substitutes for a given batch of instances.

        Args:
            sentences: list of contexts
            target_ids: list of target indexes
            target_pos: list of target word pos tags
            return_probs: return substitute probabilities if True
            target_lemmas: list of target lemmas

        Returns:
            substitutes, vocabulary and optionally substitute probabilities
        """
        probs, word2id = self.get_probs(sentences, target_ids, target_pos)

        substitutes = self.substitutes_from_probs(
            probs, word2id, sentences, target_ids, target_pos, target_lemmas
        )

        # TODO: fix the following error by recomputing probabilities
        if self.substitute_handler is not None and return_probs == True:
            raise ValueError("Probabilities might be incorrect because of lemmatization in substitute_handler")

        if return_probs:
            return substitutes, word2id, probs

        return substitutes, word2id


    def get_model_score(self,sentences: List[List[str]] = None,pred_substitutes: List[str]=None,target_ids: List[list[int]]=None):
        # from base_estimator
        return self.prob_estimator.get_model_score(sentences=sentences,pred_substitutes=pred_substitutes,target_ids=target_ids) 

    def remove_special_tokens(self,pred_substitutes:List[List[str]]):
        return self.prob_estimator.remove_special_tokens(pred_substitutes)
    
    def get_ordered_synonyms(self,original_word:str,synonyms_from_wordnet: List[str]):
        return self.prob_estimator.get_ordered_synonyms(original_word,synonyms_from_wordnet)
    

    # 实验一:only target position
    def getTargetPositionConcatScore(self,sentences: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[list[int]]=None):
        return self.prob_estimator.getTargetPositionConcatScore(sentences,pred_substitutes,target_ids)
    
    # 实验二:ig
    def getOtherPositionConcatScoreByIg(self,sentences: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[list[int]]=None):
        return self.prob_estimator.getOtherPositionConcatScoreByIg(sentences,pred_substitutes,target_ids)

    # 实验三：attention
    def getOtherPositionConcatScoreByAtten(self,sentences: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[list[int]]=None):
        return self.prob_estimator.getOtherPositionConcatScoreByAtten(sentences,pred_substitutes,target_ids)
    # 实验四：weight=1
    def getOtherPositionConcatScoreByOne(self,sentences: List[List[str]] = None,pred_substitutes: List[List[str]]=None,target_ids: List[list[int]]=None):
        return self.prob_estimator.getOtherPositionConcatScoreByOne(sentences,pred_substitutes,target_ids)
    
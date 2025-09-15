'''

'''

from typing import List, Dict, Tuple
import numpy as np
from overrides import overrides
from wordfreq import word_frequency
from lexsubgen.utils.params import build_from_dict_config
from lexsubgen.prob_estimators.base_estimator import BaseProbEstimator

# 用于组合多个模型的预测结果。
class Combiner(BaseProbEstimator):
    def __init__(self, prob_estimators: List[BaseProbEstimator], verbose: bool = False,weights:str=None,
                 stratagy_input_embedding:str=None):
        """
        Class that combines predictions from several probability estimators.
        Args:
            prob_estimators: list of probability estimators
            verbose: output verbosity
        """
        super(Combiner, self).__init__(verbose=verbose,weights=weights,stratagy_input_embedding=stratagy_input_embedding)

        self.prob_estimators = prob_estimators

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines several log-probs into one distribution.
        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary
        Returns:
            log-probs with rows representing probability distribution over vocabulary.
        """
        raise NotImplementedError()

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes log-probs over vocabulary for a given instances.
        Args:
            tokens_lists: list of contexts.
            target_ids: list of target word ids.
        Returns:
            log-probs over vocabulary and respective vocabulary.
        """
        # print(f"estimator的类型：{type(self.prob_estimators)}{self.prob_estimators}")——配置文件的类型
        # estimator这里也是dict类型，需要转一下，参考pre_processing    且estimator是elmo的实例化对象
        # {'add_bias': False,
        # 'class_name': 'lexsubgen.prob_estimators.elmo_estimator.ElmoProbEstimator',
        # 'cuda_device': 0, 'cutoff_vocab': 150000, 'direction': 'forward',
        # 'embedding_similarity': False, 'model_name': 'elmo-en', 'temperature': 1}

        # estimator_outputs = [
        #     estimator.get_log_probs(tokens_lists, target_ids)
        #     for estimator in self.prob_estimators
        # ]
        estimator_outputs=[]    # self.prob_estimators看配置文件里有多少个就有多少个
        for estimator in self.prob_estimators:
            estimator_obj, _ = build_from_dict_config(estimator)
            estimator_outputs.append(estimator_obj.get_log_probs(tokens_lists, target_ids,target_pos_tag))
        # 两个或者多个estimator配置产生的结果构成list，list[2]
        estimator_log_probs = [log_probs for log_probs, _ in estimator_outputs]
        word2ids = [w2id for _, w2id in estimator_outputs]
        estimator_log_probs, word2id = self._intersect_vocabs(
            estimator_log_probs, word2ids
        )
        combined_log_probs = self.combine(estimator_log_probs, word2id)
        return combined_log_probs, word2id
# 将多个模型的词汇表截断为它们的交集，并相应地截断模型的对数概率分布。这样做的目的是确保不同模型在相同的词汇子集上进行处理，便于后续的比较和分析。
    @staticmethod
    def _intersect_vocabs(
        log_probs_list: List[np.ndarray], word2ids: List[Dict[str, int]]
    ):
        """
        Truncates vocabularies to their intersection and respectively
        truncates model distributions.

        Args:
            log_probs_list: list of log-probs that are given by several models.
            word2ids: list of model vocabularies

        Returns:
            truncated log-probs and vocabulary
        """

        if all(x == word2ids[0] for x in word2ids[1:]):
            # All word2ids elements are the same
            return log_probs_list, word2ids[0]

        common_vocab = set(word2ids[0].keys())
        for word2id in word2ids[1:]:
            common_vocab = common_vocab.intersection(set(word2id.keys()))

        cutted_word2id = {w: idx for w, idx in word2ids[0].items() if w in common_vocab}
        words = list(cutted_word2id.keys())
        new_word2id = {w: i for i, w in enumerate(words)}

        new_probs = []
        for log_probs, word2id in zip(log_probs_list, word2ids):
            idxs = np.array([word2id[w] for w in words])
            new_probs.append(log_probs[:, idxs])
        return new_probs, new_word2id
    
    @overrides
    def get_ordered_synonyms(self,original_word:str,synonyms_from_wordnet: List[str]):
        estimator=self.prob_estimators[0]
        estimator_obj, _ = build_from_dict_config(estimator)
        return estimator_obj.get_ordered_synonyms(original_word,synonyms_from_wordnet)


# 计算所有模型的对数概率的平均值来组合预测结果。
class AverageCombiner(Combiner):
    """
    Combiner that returns average over log-probs
    """

    @overrides
    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines several log-probs into one distribution.
        Takes average over log-probs.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing
            probability distribution over vocabulary.
        """
        mean_log_probs = np.mean(log_probs, axis=0)
        return mean_log_probs

# 结合了模型预测和嵌入相似度分数。它使用一个特定的公式，其中包括了词频基础的先验分布和嵌入相似度分数。先验分布是通过 word_frequency() 函数计算的。
class BcombCombiner(Combiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        k: float = 4.0,
        s: float = 1.05,
        beta: float = 0.0,
        verbose: bool = False,
        weights:str=None,
        stratagy_input_embedding:str=None
    ):
        """
        Combines models predictions with the log-probs that comes from
        embedding similarity scores according to the formula
        :math:`P(w|C, T) \\propto \\displaystyle \\frac{P(w|C)P(w|T)}{P(w)^\\beta}`,
        where :math:`\\beta` -- is a parameter controlling how we penalize frequent words and
        :math:`P(w) = \\displaystyle \\frac{1}{(k + \\text{rank}(w))^s}`.
        For more details see N. Arefyev et al. "Combining Lexical Substitutes in Neural Word Sense Induction".
        Args:
            prob_estimators: list of probability estimators to be combined
            k: value of parameter k in prior word distribution
            s: value of parameter s in prior word distribution
            beta: value of parameter beta
            verbose: whether to output misc information
        """
        super(BcombCombiner, self).__init__(
            prob_estimators=prob_estimators, verbose=verbose,weights=weights,stratagy_input_embedding=stratagy_input_embedding)
        self.k = k
        self.s = s
        self.beta = beta
        self.bcomb_prior_log_prob = None
        self.prev_word2id = {}
    # 将不同模型的预测结果（以对数概率 log_probs 形式表示）与嵌入相似度得分进行合并处理。当传入三个对数概率数组时，
    # 会将前两个视为前向和后向传播的结果，第三个视为嵌入相似度得分，这种处理方式常用于像 ELMo 这样的模型。
    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines model predictions with embeddings similarity scores.
        If three log-probs are given this method handles first two as
        forward and backward passes and the third one as embedding similarity scores,
        this type of computation is used, for example, with ELMo model.
        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing probability distribution over vocabulary.
        """
        if self.bcomb_prior_log_prob is None or self.prev_word2id != word2id:
            self.bcomb_prior_log_prob = self.get_prior_log_prob(word2id)
        self.prev_word2id = word2id
        log_probs, similarity_log_probs = self.get_parts(log_probs)
        assert len(word2id) == log_probs.shape[-1]
        if self.beta != 0:
            log_probs -= self.beta * self.bcomb_prior_log_prob
        log_probs += similarity_log_probs
        return log_probs
    # 计算先验概率
    def get_prior_log_prob(self, word2id: Dict[str, int]) -> np.ndarray:
        """
        Get prior word distribution log-probs.
        Args:
            word2id: vocabulary
        Returns:
            `numpy.ndarray` of prior log-probs
        """
        prior_prob = np.zeros(len(word2id), dtype=np.float32)
        for word, idx in word2id.items():
            prior_prob[idx] = word_frequency(word, "en")
        idxs = prior_prob.argsort()     # 根据word_frequency的大小返回排序后的索引。[2，1，0]   index代表word 2 id的id
        prior_prob[idxs] = np.arange(len(prior_prob), 0, -1) + self.k
        return -np.log(prior_prob)[np.newaxis, :] * self.s

    @staticmethod
    def get_parts(log_probs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split log-probs onto model predictions and similarity scores.
        If length of log_probs equals 3 than first two log-probs are
        considered as forward and backward passes (used with ELMo model).
        Args:
            log_probs: list of log-probs
        Returns:
            model predictions and similarity scores.
        """
        n_dists = len(log_probs)

        if n_dists == 3:
            fwd_log_probs, bwd_log_probs, sim_log_probs = log_probs
            log_probs = fwd_log_probs + bwd_log_probs
        elif n_dists == 2:
            log_probs, sim_log_probs = log_probs    # 相似度得分，但是bert_embs.jsonnet中参数就两个，模型还一样。
        else:
            raise ValueError("Bcomb supports combination of 2 or 3 distributions!")
        return log_probs, sim_log_probs


# BcombCombiner 的一个变体，【只是】增加了一个温度参数，用于缩放相似度分数。它使用一个公式将前向和反向通过的对数概率（例如来自 ELMo 模型的结果）与嵌入的相似度分数组合。
# 调整相似度打分的平滑度
class Bcomb3Combiner(BcombCombiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        k: float = 4.0,
        s: float = 1.05,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        """
        Combines models predictions with the log-probs that comes from
        embedding similarity scores according to the formula
        :math:`P(w|C, T) \\propto \\displaystyle \\frac{P(w|C)P(w|T)}{P(w)^\\beta}`,
        where :math:`\\beta` equals to (n-1) where n -- number of estimators to combine and
        :math:`P(w) = \\displaystyle \\frac{1}{(k + \\text{rank}(w))^s}`.
        For more details see N. Arefyev et al. "Combining Lexical Substitutes in Neural Word Sense Induction".

        Args:
            prob_estimators: list of probability estimators to be combined
            k: value of parameter k in prior word distribution
            s: value of parameter s in prior word distribution
            verbose: whether to output misc information
        """
        super(Bcomb3Combiner, self).__init__(
            prob_estimators=prob_estimators, k=k, s=s, verbose=verbose
        )
        self.temperature = temperature

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines model predictions with embeddings similarity scores.
        If three log-probs are given this method handles first two as
        forward and backward passes and the third one as embedding similarity scores,
        this type of computation is used, for example, with ELMo model.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing probability distribution over vocabulary.
        """
        if self.bcomb_prior_log_prob is None or self.prev_word2id != word2id:
            self.bcomb_prior_log_prob = self.get_prior_log_prob(word2id)
        self.prev_word2id = word2id

        n_dists = len(log_probs)
        log_probs, similarity_log_probs = self.get_parts(log_probs)
        log_probs -= (n_dists - 1) * self.bcomb_prior_log_prob
        log_probs += similarity_log_probs / self.temperature
        return log_probs


class Bcomb3ZipfCombiner(BcombCombiner):
    """
        Bcomb3Combiner that uses Zipf's distribution as a prior
        word distribution. It's supposed that more frequent words
        are at the top of the vocabulary
    """

    def get_prior_log_prob(self, word2id):
        """
        Get Zipf's distribution of given size.

        Args:
            shape: size of the distribution

        Returns:
            Zipf's distribution of size `shape`
        """
        return -self.s * np.log(np.arange(len(word2id)) + self.k)[np.newaxis, :]


class BcombLmsCombiner(Combiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        alpha: float = 1.0,
        beta: float = 1.0,
        verbose: bool = False,
    ):
        """
        Combines two models distributions into one according to the formula:

        :math:`P(w|M_1, M_2) \\propto \\displaystyle \\frac{(P(w|M_1)P(w|M_2))^\\alpha}{P(w)^\\beta}` and
        :math:`P(w) = \\displaystyle \\frac{1}{1 + \\text{rank}(w)}` is a prior word distribution. It's supposed
        that words are sorted in vocabulary by their frequency -- more frequent words come first.

        Args:
            prob_estimators: list of probability estimators, supports only two estimators
            alpha: value of parameter alpha
            beta: value of parameter beta
            verbose: whether to print misc information
        """
        super(BcombLmsCombiner, self).__init__(
            prob_estimators=prob_estimators, verbose=verbose
        )
        self.alpha = alpha
        self.beta = beta

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines two model prediction into one distribution.
        Use it only for ELMo model, because its vocabulary contains words in sorted order
        Args:
            log_probs: list of log-probs from two models
            word2id: vocabulary

        Returns:
            log-probs with rows representing probability distribution over vocabulary.
        """
        assert len(log_probs) == 2
        fwd_log_probs, bwd_log_probs = log_probs
        log_probs = (fwd_log_probs + bwd_log_probs) * self.alpha

        x, y = log_probs.shape

        # [1.0, 0.5, 0.333, 0.25, ...]
        rank_w = 1.0 / (np.arange(1, y + 1))

        # Repeating rank_w x times
        # [
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     ...
        # ]
        zipf = rank_w[np.newaxis, :].repeat(x, axis=0)

        return log_probs - np.log(zipf) * self.beta

'''
用于词形还原（lemmatization）的函数
    词形还原（Lemmatization）是自然语言处理（NLP）中的一种技术，旨在将单词的不同形态转换为其基本形式，即词元（lemma）。
    与词干提取（stemming）不同，词形还原考虑了单词的词性和上下文信息，确保转换后的词形在语法上是有效的。
    dogs->dog   running->run    better->good
'''
import re
import warnings
from collections import defaultdict
from multiprocessing import cpu_count
from typing import List, Dict, Tuple, Union
import os
import numpy as np
import spacy
import cupy
import pymorphy2
from nltk.stem import WordNetLemmatizer
from spacy.lang.en import English
from tqdm import tqdm

from lexsubgen.utils.register import memory
from lexsubgen.utils.wordnet_relation import to_wordnet_pos


# model_path="66_lexsubFormyself/lexsubgen/utils/en_core_web_sum/en_core_web_sm/en_core_web_sm-3.8.0"


# 用字典缓存模型,避免重复导入且使用gpu
_nlp_cache = {}

def get_spacy_model(model_name="en_core_web_sm", use_gpu=False):
    global _nlp_cache
    # tp=cupy.cuda.runtime.getDeviceCount()  # 应该输出 > 0
    if model_name not in _nlp_cache:
        if use_gpu:
            spacy.require_gpu()
        _nlp_cache[model_name] = spacy.load(model_name)
    return _nlp_cache[model_name]



# 将某些词性标记转换为 Spacy 所使用的词性标记。——注意：swords的不同
to_spacy_pos = {
    "n": "NOUN",
    "a": "ADJ",
    "v": "VERB",
    "r": "ADV",
    "a.n": "NOUN",
    "n.v": "VERB",
    "n.a": "ADJ",
    "J": "ADJ",
    "V": "VERB",
    "R": "ADV",
    "N": "NOUN",
    # swords的词型
    "VERB": "VERB",
    "NOUN": "NOUN",
    "ADJ": "ADJ",
    "ADV": "ADV",
}

# 使用 Spacy 词形还原器对单词序列进行词形还原。spacy 参数，报错，新版本
@memory.cache
def spacy_lemmatize(
    unlem: List[str],
    pos_tag: Union[str, List[str]] = "NOUN",
    verbose: bool = False,
    spacy_version: str = spacy.__version__,
) -> List[str]:
    """
    Lemmatize sequence of words with Spacy lemmatizer.

    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information
        spacy_version: it is necessary to save cache for each version of spacy

    Returns:
        sequence of lemmatized words
    """
    if spacy_version != "2.1.8":
        warnings.warn(f"Your results may depend on the version of spacy: {spacy_version}")

    pattern = re.compile(r"[#\[-]")
    lemmatizer = English.Defaults.create_lemmatizer()
    # lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP)
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    if isinstance(pos_tag, str):
        pos_tag = [to_spacy_pos.get(pos_tag, "NOUN")] * len(unlem)
    else:
        pos_tag = [to_spacy_pos.get(pos_tag_, "NOUN") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer(word, pos_tag_)[0]
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab

# 使用旧版 Spacy 管道对单词序列进行词形还原。
@memory.cache
def old_spacy_lemmatize(
    unlem: List[str],
    verbose: bool = False,
    spacy_version: str = spacy.__version__,
) -> List[str]:
    """
    Lemmatize sequence of words with Spacy pipeline.

    Args:
        unlem: sequence of unlemmatized words
        verbose: whether to print misc information
        spacy_version: it is necessary to save cache for each version of spacy
    Returns:
        sequence of lemmatized words
    """
    # if spacy_version != "2.1.8":
    #     warnings.warn(f"Your results may depend on the version of spacy: {spacy_version}")

    # nlp = spacy.load("en", disable=["ner", "parser"])
    # tp=os.getcwd()  # config/data/datasetreaders
    # nlp = spacy.load(r"/home/ustc_zyhu/LexSub/lexsubgen/utils/en_core_web_sm/en_core_web_sm-3.8.0")  # 需要下载,直接本地

    # 使用
    nlp = get_spacy_model(r"/home/ustc_zyhu/LexSub/lexsubgen/utils/en_core_web_sm/en_core_web_sm-3.8.0", use_gpu=False)
    # doc = nlp("This is a test.")
    
    # nlp = spacy.load("./en_core_web_sm")  # 需要下载,直接本地
    # nlp=spacy.load(model_path)

    lemmatized_words = []

    with warnings.catch_warnings(record=True) as wn:
        # When using spacy 2.1.8 it warns: "DeprecationWarning: [W016] The keyword argument `n_threads` is now deprecated.
        #   As of v2.2.2, the argument `n_process` controls parallel inference via multiprocessing."
        warnings.simplefilter("ignore")
        gen = zip(nlp.pipe(unlem, batch_size=1000, n_process=cpu_count()), unlem)

        if verbose:
            # 执行多次
            gen = tqdm(gen, total=len(unlem), desc=f"Lemmatization of {len(unlem)} words")

        for spacyed, word in gen:
            if "#" in word or "[" in word or word == "":
                lemma = word
            else:
                lemma = (
                    spacyed[0].lemma_
                    if spacyed[0].lemma_ != "-PRON-"
                    else spacyed[0].lower_
                )
            lemmatized_words.append(lemma)

        # Checking if it is not just DeprecationWarning
        assert len(wn) <= 1, str(wn)
        assert len(wn) <= 1 or issubclass(wn[-1].category, DeprecationWarning), str(wn)
        assert len(wn) <= 1 or "deprecated" in str(wn[-1].message), str(wn)

    return lemmatized_words

# 使用 NLTK 词形还原器对单词序列进行词形还原。
@memory.cache
def nltk_lemmatize(
    unlem: List[str], pos_tag: Union[str, List[str]] = "n", verbose: bool = False
) -> List[str]:
    """
    Lemmatize sequence of words with nltk tokenizer.

    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words

    """
    pattern = re.compile(r"[#\[-]")
    lemmatizer = WordNetLemmatizer()
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    # convert to appropriate pos abbreviation
    if isinstance(pos_tag, str):
        pos_tag = [to_wordnet_pos.get(pos_tag, "n")] * len(unlem)
    else:
        pos_tag = [to_wordnet_pos.get(pos_tag_, "n") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer.lemmatize(word, pos_tag_)
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab

# 使用 Pymorphy2 词形还原器对俄语单词序列进行词形还原。
@memory.cache
def pymorphy_ru_lemmatize(
    unlem: List[str],
    verbose: bool = False,
    pymorphy_version: str = pymorphy2.__version__,
) -> List[str]:
    """
    Lemmatizes sequence of words with Pymorphy lemmatizer.

    Args:
        unlem: sequence of unlemmatized words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """
    lemmatizer = pymorphy2.MorphAnalyzer()
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc='Vocabulary Lemmatization')

    new_vocab = [
        word if ('#' in word or '[' in word)
        else lemmatizer.parse(word)[0].normal_form
        for word in gen
    ]

    return new_vocab

# 根据指定的词形还原器名称选择合适的词形还原器对单词序列进行词形还原。
def lemmatize_words(
    unlem: List[str],
    lemmatizer_name: str,
    pos_tag: Union[str, List[str]] = "n",
    verbose: bool = False,
) -> List[str]:
    """
    This function just chooses right lemmatizer that is specified by name.

    Args:
        unlem: sequence of unlemmatized words
        lemmatizer_name: name of the lemmatizer (currently supported lemmatizers are nltk and Spacy).
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """
    if lemmatizer_name == "nltk":
        lemmatized = nltk_lemmatize(unlem, pos_tag, verbose)
    elif lemmatizer_name == "spacy":
        lemmatized = spacy_lemmatize(unlem, pos_tag, verbose)
    elif lemmatizer_name == "spacy_old":
        lemmatized = old_spacy_lemmatize(unlem, verbose)
    elif lemmatizer_name == "pymorphy-ru":
        lemmatized = pymorphy_ru_lemmatize(unlem, verbose)
    else:
        raise ValueError(f"Incorrect lemmatizer type: {lemmatizer_name}")
    return lemmatized

# 将不同词形的概率聚合到它们的词元上。支持两种聚合策略：取最大值（"max"）和求和（"sum"）。
def lemmatize_batch(
    probs: np.ndarray,
    forms_ids_lists: List[List[int]],
    strategy: str = "max",
    parallel: bool = False,
) -> np.ndarray:
    """
    Aggregates probabilities of different word forms to their lemmas.
    Different aggregation strategies could be chosen, currently we support
    taking maximum probability of all word forms and summing them.

    Args:
        probs: matrix of distributions over vocabulary for each batch instance
        forms_ids_lists: list of indexes of word forms for a lemma
        strategy: aggregation strategy (max or sum)
        parallel: whether to aggregate data for different words in parallel (default: False)

    Returns:
        new probability distributions with aggregate probabilities for lemmas
    """
    assert strategy == "max" or strategy == "sum"
    new_batch = np.zeros((probs.shape[0], 0))
    if not parallel:
        new_batches = []
        if forms_ids_lists:
            for i, forms_ids in enumerate(forms_ids_lists):
                new_batches.append(
                    np.__getattribute__(strategy)(
                        probs[:, forms_ids], axis=1, keepdims=True
                    )
                )
            new_batch = np.concatenate(new_batches, axis=1)
    else:
        new_batch = probs[:, forms_ids_lists].__getattribute__(strategy)(axis=-1)
    return new_batch

# 使用指定的词形还原器对词汇表进行词形还原，将原始词汇表缩小为词元词汇表。
@memory.cache
def get_all_vocabs(
    old_word2id: Dict[str, int],
    lemmatizer: str,
    pos_tag: str = "n",
    verbose: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Method that lemmatizes a vocabulary with the chosen lemmatizer.
    So the original vocabulary shrinks to vocabulary of their lemmas.

    Args:
        old_word2id: old vocabulary with unlemmatized words
        lemmatizer: name of the lemmatizer to be used for processing
        verbose: whether to print misc information

    Returns:
        mapping from lemmas to their word forms,
        mapping from words to indexes (new vocabulary with lemmatized words)
    """
    sorted_vocab = sorted(old_word2id.items(), key=lambda x: x[0])
    sorted_words, sorted_idxs = list(zip(*sorted_vocab))

    new_vocab = lemmatize_words(sorted_words, lemmatizer, pos_tag, verbose)

    lemma2words = defaultdict(list)
    word2id = dict()
    for word, old_idx, lemma in zip(sorted_words, sorted_idxs, new_vocab):
        lemma2words[lemma].append(old_idx)
        word2id[lemma] = word2id.get(lemma, len(word2id))

    return lemma2words, word2id


@memory.cache
def get_wordform2lemma(
    vocabulary: List[str],
    lemmatizer: str,
    pos_tag: str = "n",
    verbose: bool = False
) -> Dict[str, str]:
    """
    Wordform2lemma is a dict that maps word forms to its lemmas
    Args:
        vocabulary: vocabulary of word forms
        lemmatizer: name of the lemmatizer to be used for processing
        pos_tag: part of speech that will be used in lemmatization
        verbose: whether to print misc information

    Returns: mapping from word form to its lemma
    """
    lemmatized = lemmatize_words(vocabulary, lemmatizer, pos_tag, verbose)
    return dict(zip(vocabulary, lemmatized))

@memory.cache
def lemmatize_candidates(
    target_ids: List[int],
    sentences: List[List[str]],
    candidate_lists: List[List[str]],
    lemmatizer: str = "spacy",
    verbose: bool = False
) -> List[List[str]]:
    """
    Inflect candidate words to match the form and POS of the target word in each sentence.

    Args:
        target_ids: list of indices of target words in sentences
        sentences: list of tokenized sentences
        candidate_lists: list of candidate word lists corresponding to each sentence
        lemmatizer: 'spacy' or 'nltk'
        verbose: whether to print progress

    Returns:
        List of lists of inflected candidate words
    """
    assert len(target_ids) == len(sentences) == len(candidate_lists)

    inflected_all = []

    if lemmatizer == "spacy":
        nlp = get_spacy_model(r"/home/ustc_zyhu/LexSub/lexsubgen/utils/en_core_web_sm/en_core_web_sm-3.8.0", use_gpu=False)

        for i, (sent, target_idx, candidates) in enumerate(zip(sentences, target_ids, candidate_lists)):
            if verbose:
                print(f"Processing sentence {i}: target word index {target_idx}")

            target_word = sent[target_idx]
            doc = nlp(" ".join(sent))
            target_token = doc[target_idx]
            target_tag = target_token.tag_  # 详细词性标签
            inflected_candidates = []

            for cand in candidates:
                cand_doc = nlp(cand)
                cand_token = cand_doc[0]
                # 尝试匹配目标词形态
                try:
                    inflected = cand_token._.inflect(target_tag) if hasattr(cand_token._, "inflect") else cand
                except Exception:
                    inflected = cand
                inflected_candidates.append(inflected or cand)

            inflected_all.append(inflected_candidates)

    elif lemmatizer == "nltk":
        from nltk.stem import WordNetLemmatizer
        from lexsubgen.utils.wordnet_relation import to_wordnet_pos

        lemmatizer_nltk = WordNetLemmatizer()
        for sent, target_idx, candidates in zip(sentences, target_ids, candidate_lists):
            target_word = sent[target_idx]
            target_pos = to_wordnet_pos.get("n", "n")  # 默认名词，可扩展
            inflected_candidates = [
                lemmatizer_nltk.lemmatize(cand, target_pos) for cand in candidates
            ]
            inflected_all.append(inflected_candidates)

    else:
        raise ValueError(f"Unsupported lemmatizer: {lemmatizer}")

    return inflected_all

import json
import os
from string import punctuation
from typing import NoReturn, Dict, List, Tuple

import numpy as np
import torch
from overrides import overrides
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
import torch.nn.functional as F
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from captum.attr import LayerIntegratedGradients


class Debertav3ProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "mpnet-base",
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        cuda_device: int = -1,
        sim_func: str = "dot-product",
        unk_word_embedding: str = "first_subtoken",
        filter_vocabulary_mode: str = "none",
        verbose: bool = False,
    ):
        """
        Probability estimator based on the Roberta model.
        See Y. Liu et al. "RoBERTa: A Robustly Optimized
        BERT Pretraining Approach".

        Args:
            mask_type: the target word masking strategy.
            model_name: Roberta model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            unk_word_embedding: how to handle words that are splitted into multiple subwords when computing
            embedding similarity
            verbose: whether to print misc information
        """
        super(Debertav3ProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.unk_word_embedding = unk_word_embedding
        self.filter_vocabulary_mode = filter_vocabulary_mode
        self.prev_word2id = {}

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "roberta",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "unk_word_embedding": self.unk_word_embedding,
            }
        }

        self.register_model()

        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

    @property
    def tokenizer(self):
        """
        Model tokenizer.

        Returns:
            `transformers.RobertaTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]

    @property
    def parameters(self):
        parameters = f"{self.mask_type}{self.model_name}" \
                     f"{self.use_attention_mask}{self.filter_vocabulary_mode}"

        if self.embedding_similarity:
            parameters += f"embs{self.unk_word_embedding}{self.sim_func}"

        return parameters

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in Debertav3ProbEstimator.loaded:
            self.model_name = '/home/ustc_zyhu/LexSub/transformer_models/'+self.model_name
            roberta_model = DebertaV2ForMaskedLM.from_pretrained(self.model_name)
            roberta_model.to(self.device).eval()
            roberta_tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
            roberta_word2id = Debertav3ProbEstimator.load_word2id(roberta_tokenizer)
            filter_word_ids = Debertav3ProbEstimator.load_filter_word_ids(
                roberta_word2id, punctuation
            )
            word_embeddings = (
                roberta_model.deberta.embeddings.word_embeddings.weight.detach()
            )

            # norms = np.linalg.norm(word_embeddings, axis=-1, keepdims=True)
            # normed_word_embeddings = word_embeddings / norms

            Debertav3ProbEstimator.loaded[self.model_name] = {
                "model": roberta_model,
                "tokenizer": roberta_tokenizer,
                "embeddings": word_embeddings,
                "normed_embeddings": word_embeddings,
                "word2id": roberta_word2id,
                "filter_word_ids": filter_word_ids,
            }
            Debertav3ProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            Debertav3ProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @property
    def normed_embeddings(self) -> np.ndarray:
        """
        Attribute that acquires model word normed_embeddings.

        Returns:
            2-D `numpy.ndarray` with rows representing word vectors.
        """
        return self.loaded[self.model_name]["normed_embeddings"]

    def get_emb_similarity(
        self, tokens_batch: List[List[str]], target_ids_batch: List[int],
    ) -> np.ndarray:
        """
        Computes similarity between each target word and substitutes
        according to their embedding vectors.

        Args:
            tokens_batch: list of contexts
            target_ids_batch: list of target word ids in the given contexts

        Returns:
            similarity scores between target words and
            words from the model vocabulary.
        """
        if self.sim_func == "dot-product":
            embeddings = self.embeddings
        else:
            embeddings = self.normed_embeddings

        target_word_embeddings = []
        for tokens, pos in zip(tokens_batch, target_ids_batch):
            tokenized = self.tokenize_around_target(tokens, pos, self.tokenizer)
            _, _, target_subtokens_ids = tokenized

            target_word_embeddings.append(
                self.get_target_embedding(target_subtokens_ids, embeddings)
            )

        target_word_embeddings = np.vstack(target_word_embeddings)
        emb_sim = np.matmul(target_word_embeddings, embeddings.T)

        return emb_sim / self.temperature

    def get_target_embedding(
        self,
        target_subtokens_ids: List[int],
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Returns an embedding that will be used if the given word is not presented
        in the vocabulary. The word is split into subwords and depending on the
        self.unk_word_embedding parameter the final embedding is built.

        Args:
            word: word for which the vector should be given
            target_subtokens_ids: vocabulary indexes of target word subtokens
            embeddings: roberta embeddings of target word subtokens

        Returns:
            embedding of the unknown word
        """
        if self.unk_word_embedding == "mean":
            return embeddings[target_subtokens_ids].mean(axis=0, keepdims=True)
        elif self.unk_word_embedding == "first_subtoken":
            return embeddings[target_subtokens_ids[0]]
        elif self.unk_word_embedding == "last_subtoken":
            return embeddings[target_subtokens_ids[-1]]
        else:
            raise ValueError(
                f"Incorrect value of unk_word_embedding: "
                f"{self.unk_word_embedding}"
            )

    @staticmethod
    def load_word2id(tokenizer: DebertaV2Tokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.

        Args:
            tokenizer: `transformers.RobertaTokenizer` tokenizer

        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id

    @staticmethod
    def load_filter_word_ids(word2id: Dict[str, int], filter_chars: str) -> List[int]:
        """
        Gathers words that should be filtered from the end distribution, e.g.
        punctuation.

        Args:
            word2id: model vocabulary
            filter_chars: words with this chars should be filtered from end distribution.

        Returns:
            Indexes of words to be filtered from the end distribution.
        """
        filter_word_ids = []
        set_filter_chars = set(filter_chars)
        for word, idx in word2id.items():
            if len(set(word) & set_filter_chars):
                filter_word_ids.append(idx)
        return filter_word_ids

    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.

        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["filter_word_ids"]

    @staticmethod
    def tokenize_around_target(
        tokens: List[str],
        target_idx: int,
        tokenizer: DebertaV2Tokenizer = None,
    ):
        left_specsym_len = 1  # for BERT / ROBERTA there is 1 spec token before text
        input_text = ' '.join(tokens)
        tokenized_text = tokenizer.encode(' ' + input_text, add_special_tokens=True)

        left_ctx = ' '.join(tokens[:target_idx])
        target_start = left_specsym_len + len(tokenizer.encode(
            ' ' + left_ctx, add_special_tokens=False
        ))

        left_ctx_target = ' '.join(tokens[:target_idx + 1])
        target_subtokens_ids = tokenizer.encode(
            ' ' + left_ctx_target, add_special_tokens=False
        )[target_start - left_specsym_len:]

        return tokenized_text, target_start, target_subtokens_ids

    def prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: DebertaV2Tokenizer = None,
    ):
        if tokenizer is None:
            tokenizer = self.tokenizer

        roberta_batch_of_tokens, roberta_batch_of_target_ids = [], []
        max_seq_len = 0
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            tokenized = self.tokenize_around_target(tokens, target_idx, tokenizer)
            context, target_start, target_subtokens_ids = tokenized

            if self.mask_type == "masked":
                context = context[:target_start] + \
                          [tokenizer.mask_token_id] + \
                          context[target_start + len(target_subtokens_ids):]
            elif self.mask_type != "not_masked":
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")

            if len(context) > 512:
                first_subtok = context[target_start]
                # Cropping maximum context around the target word
                left_idx = max(0, target_start - 256)
                right_idx = min(target_start + 256, len(context))
                context = context[left_idx: right_idx]
                target_start = target_start if target_start < 256 else 255
                assert first_subtok == context[target_start]

            max_seq_len = max(max_seq_len, len(context))

            roberta_batch_of_tokens.append(context)
            roberta_batch_of_target_ids.append(target_start)

        assert max_seq_len <= 512

        input_ids = np.vstack([
            tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
            for tokens in roberta_batch_of_tokens
        ])

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids, roberta_batch_of_target_ids

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int],
    ) -> np.ndarray:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        input_ids, mod_target_ids = self.prepare_batch(tokens_lists, target_ids)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
            attention_mask = attention_mask.float().to(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = np.vstack([
                logits[idx, target_idx, :].cpu().numpy()
                for idx, target_idx in enumerate(mod_target_ids)
            ])
            return logits

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".

        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """

        if self.embedding_similarity:
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids)

        return logits, self.word2id
    
    # 实验 N 分数：是 rerank 
    # 实验一分数：concatenate 目标位置处(若分词，需要平均后concatenate) version 1: average after concatenate
    def getTargetPositionConcatScore(self, tokens_lists = None, pred_substitutes = None, target_ids = None):
        """
        Args:
            tokens_lists: list of list of str, 每个元素是一句话的token序列
            pred_substitutes: list of list of str, 每个元素是候选替换词
            target_ids: list of int, 每个元素是目标词在句子中的位置

        Returns:
            batch_scores: list，每句话的候选词相似度列表
        """
        start_layer = 3
        end_layer = -2   # python 切片里支持负索引，-2 表示倒数第2层


        # 词形还原候选词
        # pred_substitutes = lemmatize_candidates(target_ids, tokens_lists, pred_substitutes)

        batch_scores = []

        for tokens, substitutes, target_id in zip(tokens_lists, pred_substitutes, target_ids):
            orig_word = tokens[target_id]

            # 构造替换后的句子集合（包括原句）
            sentences = []
            sentences.append(tokens.copy())  # 原句
            for sub in substitutes:
                new_tokens = tokens.copy()
                new_tokens[target_id] = sub
                sentences.append(new_tokens)

            # step1: tokenizer 编码
            encodings = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,  # 超过 max_length 会截断
                max_length=512,
                is_split_into_words=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings,output_hidden_states=True)
            hidden_states = outputs.hidden_states  # list[num_layers](batch, seq_len, hidden_dim)

            # step3: 在 3 到 L-2 层取 hidden states，并对目标词 subwords 平均，再拼接
            concat_reps = []
            for layer_hs in hidden_states[start_layer:end_layer]:
                reps = []
                for i, tokens in enumerate(sentences):
                    target_word = tokens[target_id]  # 目标词
                    # 切片得到子词对应的输入
                    tokens_before = tokens[:target_id]       # 目标词前的子词
                    tokens_target = tokens[:target_id + 1]  # 包含目标词的子词

                    # 编码两次，得到 subword token ids, CLS的标记
                    ids_before = self.tokenizer.convert_tokens_to_ids(tokens_before)
                    ids_target = self.tokenizer.convert_tokens_to_ids(tokens_target)

                    # 目标词的 subword 索引范围
                    start_idx = len(ids_before)+1
                    end_idx = len(ids_target)+1

                    # 从 hidden_states 取对应位置的向量，并平均
                    vec = layer_hs[i, start_idx:end_idx, :].mean(dim=0)
                    reps.append(vec)

                reps = torch.stack(reps, dim=0)  # (batch, hidden_dim)
                concat_reps.append(reps)

            concat_reps = torch.cat(concat_reps, dim=-1)  # (batch, concat_dim)

            # step4: 相似度
            orig_vec = concat_reps[0]   # 原句
            subs_vecs = concat_reps[1:] # 替换句子
            sims = F.cosine_similarity(subs_vecs, orig_vec.unsqueeze(0), dim=-1)

            batch_scores.append(sims.tolist())
    
        return batch_scores
    

    # 实验二分数：concatenate 重要目标位置处 ig
    def getOtherPositionConcatScoreByIg(self, tokens_lists = None, pred_substitutes = None, target_ids = None):
        """
            Args:
                tokens_lists: list[list[str]]   批次内句子（分好词）
                pred_substitutes: list[list[str]]  每条句子的候选替代词
                target_ids: list[int]  每条句子中目标词的位置
                topk: int  选 attribution 最大的几个位置
            Return:
                results: list[dict]  每条句子结果
        """
        ig_batch=1
        # pred_substitutes = lemmatize_candidates(target_ids, tokens_lists, pred_substitutes)
        # 这里设置要取的层，比如 3 到 倒数第2层
        start_layer = 3
        end_layer = -2   # python 切片里支持负索引，-2 表示倒数第2层
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # ===== Step1: 把目标词替换成 [MASK] =====
        masked_sentences = []
        for tokens, tid in zip(tokens_lists, target_ids):
            new_tokens = tokens.copy()
            new_tokens[tid] = self.tokenizer.mask_token  # 替换成 [MASK]
            masked_sentences.append(new_tokens)

        # ===== Step1: 批次 attribution，得到每条数据的 top-k重要位置 =====
        encodings = self.tokenizer(masked_sentences,
                            is_split_into_words=True,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512).to(self.device)
        input_ids = encodings["input_ids"]

        # 找到每个句子的 [MASK] 位置
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)

        # ===== Step1: 记录每个句子的 target span =====
        target_spans = []  # [(start_pos, end_pos, target_len), ...]
        for tokens, tid in zip(tokens_lists, target_ids):
            pre_sub_len = 0
            for tok in tokens[:tid]:
                pre_sub_len += len(self.tokenizer.tokenize(tok))
            start_pos = pre_sub_len + 1  # +1 for [CLS]
            target_len = len(self.tokenizer.tokenize(tokens[tid]))
            end_pos = start_pos + target_len
            target_spans.append((start_pos, end_pos, target_len))


        # ===== 关键修改：取原目标词的 id =====
        target_word_ids = []
        for tokens, tid in zip(tokens_lists, target_ids):
            original_token = tokens[tid]  # 原目标词
            # 用 tokenizer 分词得到子词列表
            sub_tokens = self.tokenizer.tokenize(original_token)
            wid = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            target_word_ids.append(wid)
        
        # Step: 批次分割用于 IG
        top_positions_batch = []
        top_weights_batch = []
        for start in range(0, input_ids.size(0), ig_batch):
            end = start + ig_batch
            input_ids_batch = input_ids[start:end].to(self.device)
            mask_indices_batch = mask_indices[start:end]
            target_word_ids_batch = target_word_ids[start:end]


            def forward_func(input_embeds):
                outputs = self.model(inputs_embeds=input_embeds)
                logits = outputs.logits     # (batch_size, seq_len, vocab_size)
                batch_logits = []
                for b, (row ,col)in enumerate(mask_indices_batch):
                    # target_word_ids_batch[b] 是 list[int]，可能包含多个 sub-token
                    target_ids = target_word_ids_batch[b]
                     # 取每个 sub-token 对应位置的 logit，然后平均
                    sub_logits = []
                    for tid in target_ids:
                        sub_logits.append(logits[b, col, tid])
                    # 多个 sub-token 取平均
                    batch_logits.append(torch.stack(sub_logits).mean())
                    
                return torch.stack(batch_logits)

            lig = LayerIntegratedGradients(forward_func, self.model.deberta.embeddings)
            
            baseline_ids = torch.full_like(input_ids_batch, self.tokenizer.pad_token_id).to(self.device)
            
            baseline_embeds = self.model.deberta.embeddings(baseline_ids)
            input_embeds_batch = self.model.deberta.embeddings(input_ids_batch)

            attributions, _ = lig.attribute(inputs=input_embeds_batch,
                                            baselines=baseline_embeds,
                                            return_convergence_delta=True,n_steps=20)
            
            # 去除pad
            attention_mask_batch = (input_ids_batch != self.tokenizer.pad_token_id).long()
            attr_scores = attributions.sum(dim=-1)  # [batch, seq_len]
            # 屏蔽 pad 的 attribution
            attr_scores = attr_scores * attention_mask_batch


            # 每条句子选出 topk位置
            special_tokens = [0, input_ids.size(1)-1]  # CLS 和 SEP
            for i, (start_pos, end_pos, _) in enumerate(target_spans[start:end]):
                scores = attr_scores[i]

                # 截断，防止 pad 干扰
                mask = attention_mask_batch[i]  # [seq_len]
                valid_len = mask.sum().item()   # 实际 token 长度（不含 pad）
                scores = scores[:valid_len]     # 截断到真实长度

                position_score = [(pos, score.item()) for pos, score in enumerate(scores)]
                position_score.sort(key=lambda x: x[1], reverse=True)  # 从大到小排序

                # 过滤条件：位置不在目标位置和特殊符号中，且分值>0
                filtered = [
                    (pos, score) for pos, score in position_score
                    if pos not in special_tokens and not (start_pos <= pos < end_pos)
                ]
                
                top_positions = [pos for pos, score in filtered]
                top_scores = [score for pos, score in filtered]
                weights = torch.softmax(torch.tensor(top_scores), dim=0).tolist()

                top_positions_batch.append(top_positions)
                top_weights_batch.append(weights)
            torch.cuda.empty_cache()

        # ===== Step2: 替换目标词，计算原句 vs 替代句的 hidden state相似度 =====
        results = []
        
        for idx,(tokens, substitutes, tid,(start_pos, end_pos, old_len), top_positions, top_weights) in enumerate(zip(tokens_lists, pred_substitutes, target_ids, target_spans
                                                                                                                  ,top_positions_batch,top_weights_batch)):

            # 原句 hidden states
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
            orig_hid = self.model(**inputs, output_hidden_states=True)
            hs = orig_hid.hidden_states
            orig_hid = torch.cat(hs[start_layer:end_layer], dim=-1).squeeze(0)   # 拼接多层

            weight_map = {p: w for p, w in zip(top_positions, top_weights)}
            sims_per_sub = []
            for sub in substitutes:
                perturbed_tokens = tokens.copy()
                perturbed_tokens[tid] = sub
                perturbed_inputs = self.tokenizer(perturbed_tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
                perturbed_hid = self.model(**perturbed_inputs, output_hidden_states=True)
                perturbed_hs = perturbed_hid.hidden_states
                pert_hid = torch.cat(perturbed_hs[start_layer:end_layer], dim=-1).squeeze(0)

                sims = []
                # --- 新句目标 span
                # 替代词长度差，决定后续 token 的位移
                new_len = len(self.tokenizer.tokenize(sub))
                delta = new_len - old_len

                sims = []
                for p in top_positions:
                    if p < start_pos:
                        mp = p
                    elif p >= end_pos:
                        mp = p + delta
                    else:
                        continue  # 已屏蔽目标词 span

                    if 0 <= mp < pert_hid.size(0):
                        sim = F.cosine_similarity(orig_hid[p], pert_hid[mp], dim=0).item()
                        sims.append(sim * weight_map[p])

                # 目标词相似度（整体）
                # orig_target_vec = orig_hid[start_pos:end_pos].mean(dim=0)
                # pert_target_vec = pert_hid[start_pos:start_pos + new_len].mean(dim=0)
                # target_sim = F.cosine_similarity(orig_target_vec, pert_target_vec, dim=0).item()
                # sims.append(target_sim)

                sims_per_sub.append({"substitute": sub, "sims": sims})

            results.append(sims_per_sub)


        return results
    
    # attention 版本:对齐替换后的句和原句
    def getOtherPositionConcatScoreByAtten(self, tokens_lists=None, pred_substitutes=None, target_ids=None):
       
        start_layer, end_layer = 3, -2  # 取多层平均/拼接的范围

        # ===== 编码（批处理）=====
        encodings = self.tokenizer(
            tokens_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        input_ids = encodings["input_ids"]
        attention_mask_batch = encodings["attention_mask"]

        # ===== 计算每条样本中 目标词的 sub-token 起止区间 =====
        target_spans = []  # [(start_pos, end_pos, old_len), ...]  end_pos 为开区间
        for tokens, tid in zip(tokens_lists, target_ids):
            pre_sub_len = 0
            for tok in tokens[:tid]:
                pre_sub_len += len(self.tokenizer.tokenize(tok))
            start_pos = pre_sub_len + 1  # +1 for [CLS]

            target_len = len(self.tokenizer.tokenize(tokens[tid]))
            end_pos = start_pos + target_len

            target_spans.append((start_pos, end_pos, target_len))

        # ===== 用注意力找“其它 token→目标词”的权重（整序列归一化; 屏蔽CLS/SEP/PAD/目标词）=====
        top_positions_batch = []
        top_weights_batch = []
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=(input_ids != self.tokenizer.pad_token_id),
                output_attentions=True
            )
            # attentions: tuple(num_layers, B, H, L, L)  → 选层/均头
            selected = outputs.attentions[start_layer:end_layer]
            attn_scores = torch.stack(selected, dim=0).mean(dim=0).mean(dim=1)  # (B, L, L)

            for i, (start_pos, end_pos, _) in enumerate(target_spans):
                # 每个 query 位置对“目标词所有 sub-token keys”的平均注意力
                scores = attn_scores[i, :, start_pos:end_pos].mean(dim=1)  # (L,)

                # 构造保留掩码（True=保留）
                mask = attention_mask_batch[i].bool().clone()  # 有效token
                mask[0] = False  # [CLS]
                if self.tokenizer.sep_token_id is not None:
                    mask[input_ids[i] == self.tokenizer.sep_token_id] = False
                if self.tokenizer.pad_token_id is not None:
                    mask[input_ids[i] == self.tokenizer.pad_token_id] = False
                # 屏蔽目标词自身 sub-token
                mask[start_pos:end_pos] = False

                # 只对保留部分做 softmax 归一化，得到权重
                keep_pos = mask.nonzero(as_tuple=True)[0]
                keep_scores = scores[keep_pos]
                keep_probs = torch.softmax(keep_scores, dim=-1)

                top_positions_batch.append(keep_pos.tolist())
                top_weights_batch.append(keep_probs.tolist())

        # ===== 计算 原句 vs 替代句 在这些位置的相似度（做 delta 对齐）=====
        results = []
        for i, (tokens, substitutes, tid, top_positions, top_weights) in enumerate(
            zip(tokens_lists, pred_substitutes, target_ids, top_positions_batch, top_weights_batch)
        ):
            # 原句 hidden states（拼接多层）
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                orig_out = self.model(**inputs, output_hidden_states=True)
            orig_hid = torch.cat(orig_out.hidden_states[start_layer:end_layer], dim=-1).squeeze(0)  # (L, D)

            start_pos, end_pos, old_len = target_spans[i]
            weight_map = {p: w for p, w in zip(top_positions, top_weights)}

            sims_per_sub = []
            for sub in substitutes:
                # 生成替代句
                perturbed_tokens = tokens.copy()
                perturbed_tokens[tid] = sub

                pert_inputs = self.tokenizer(perturbed_tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    pert_out = self.model(**pert_inputs, output_hidden_states=True)
                pert_hid = torch.cat(pert_out.hidden_states[start_layer:end_layer], dim=-1).squeeze(0)  # (L', D)

                # 仅需替换词的 sub-token 长度即可得到位移量
                new_len = len(self.tokenizer.tokenize(sub))
                delta = new_len - old_len

                sims = []
                for p in top_positions:
                    if p < start_pos:
                        mp = p
                    elif p >= end_pos:
                        mp = p + delta
                    else:
                        # 理论上不会进来（已屏蔽目标词）
                        continue

                    if 0 <= mp < pert_hid.size(0):
                        sim = F.cosine_similarity(orig_hid[p], pert_hid[mp], dim=0).item()
                        sims.append(sim * weight_map[p])
                # ===== 新增：目标词本身的相似度 加不加？=====
                # orig_target_vec = orig_hid[start_pos:end_pos].mean(dim=0)
                # pert_target_vec = pert_hid[start_pos:start_pos + new_len].mean(dim=0)
                # target_sim = F.cosine_similarity(orig_target_vec, pert_target_vec, dim=0).item()
                # sims.append(target_sim)
                
                sims_per_sub.append({"substitute": sub, "sims": sims})

            results.append(sims_per_sub)

        return results
    
    # 所有权重均为1的版本:注意 原始目标位置
    def getOtherPositionConcatScoreByOne(self, tokens_lists=None, pred_substitutes=None, target_ids=None):
       
        start_layer, end_layer = 3, -2  # 取多层平均/拼接的范围

        # ===== 编码（批处理）=====
        encodings = self.tokenizer(
            tokens_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        input_ids = encodings["input_ids"]
        attention_mask_batch = encodings["attention_mask"]

        # ===== 计算每条样本中 目标词的 sub-token 起止区间 =====
        target_spans = []  # [(start_pos, end_pos, old_len), ...]  end_pos 为开区间
        for tokens, tid in zip(tokens_lists, target_ids):
            pre_sub_len = 0
            for tok in tokens[:tid]:
                pre_sub_len += len(self.tokenizer.tokenize(tok))
            start_pos = pre_sub_len + 1  # +1 for [CLS]

            target_len = len(self.tokenizer.tokenize(tokens[tid]))
            end_pos = start_pos + target_len

            target_spans.append((start_pos, end_pos, target_len))

        # ===== 用注意力找“其它 token→目标词”的权重（整序列归一化; 屏蔽CLS/SEP/PAD/目标词）=====
        top_positions_batch = []
        top_weights_batch = []
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=(input_ids != self.tokenizer.pad_token_id),
                output_attentions=True
            )
            # attentions: tuple(num_layers, B, H, L, L)  → 选层/均头
            selected = outputs.attentions[start_layer:end_layer]
            attn_scores = torch.stack(selected, dim=0).mean(dim=0).mean(dim=1)  # (B, L, L)

            for i, (start_pos, end_pos, _) in enumerate(target_spans):
                # 每个 query 位置对“目标词所有 sub-token keys”的平均注意力
                scores = attn_scores[i, :, start_pos:end_pos].mean(dim=1)  # (L,)

                # 构造保留掩码（True=保留）
                mask = attention_mask_batch[i].bool().clone()  # 有效token
                mask[0] = False  # [CLS]
                if self.tokenizer.sep_token_id is not None:
                    mask[input_ids[i] == self.tokenizer.sep_token_id] = False
                if self.tokenizer.pad_token_id is not None:
                    mask[input_ids[i] == self.tokenizer.pad_token_id] = False
                # 屏蔽目标词自身 sub-token
                mask[start_pos:end_pos] = False

                keep_pos = mask.nonzero(as_tuple=True)[0]
                keep_probs =torch.ones_like(keep_pos, dtype=torch.float)

                top_positions_batch.append(keep_pos.tolist())
                top_weights_batch.append(keep_probs.tolist())

        # ===== 计算 原句 vs 替代句 在这些位置的相似度（做 delta 对齐）=====
        results = []
        for i, (tokens, substitutes, tid, top_positions, top_weights) in enumerate(
            zip(tokens_lists, pred_substitutes, target_ids, top_positions_batch, top_weights_batch)
        ):
            # 原句 hidden states（拼接多层）
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                orig_out = self.model(**inputs, output_hidden_states=True)
            orig_hid = torch.cat(orig_out.hidden_states[start_layer:end_layer], dim=-1).squeeze(0)  # (L, D)

            start_pos, end_pos, old_len = target_spans[i]
            weight_map = {p: w for p, w in zip(top_positions, top_weights)}

            sims_per_sub = []
            for sub in substitutes:
                # 生成替代句
                perturbed_tokens = tokens.copy()
                perturbed_tokens[tid] = sub

                pert_inputs = self.tokenizer(perturbed_tokens, is_split_into_words=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    pert_out = self.model(**pert_inputs, output_hidden_states=True)
                pert_hid = torch.cat(pert_out.hidden_states[start_layer:end_layer], dim=-1).squeeze(0)  # (L', D)

                # 仅需替换词的 sub-token 长度即可得到位移量
                new_len = len(self.tokenizer.tokenize(sub))
                delta = new_len - old_len

                sims = []
                for p in top_positions:
                    if p < start_pos:
                        mp = p
                    elif p >= end_pos:
                        mp = p + delta
                    else:
                        # 理论上不会进来（已屏蔽目标词）
                        continue

                    if 0 <= mp < pert_hid.size(0):
                        sim = F.cosine_similarity(orig_hid[p], pert_hid[mp], dim=0).item()
                        sims.append(sim * weight_map[p])
                # ===== 新增：目标词本身的相似度 加不加？=====
                orig_target_vec = orig_hid[start_pos:end_pos].mean(dim=0)
                pert_target_vec = pert_hid[start_pos:start_pos + new_len].mean(dim=0)
                target_sim = F.cosine_similarity(orig_target_vec, pert_target_vec, dim=0).item()
                sims.append(target_sim)
                
                sims_per_sub.append({"substitute": sub, "sims": sims})

            results.append(sims_per_sub)

        return results
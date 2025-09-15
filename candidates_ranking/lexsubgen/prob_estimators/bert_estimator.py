'''
输入bert前的处理+返回概率分布
'''
import json
import os
from string import punctuation
from typing import NoReturn, Dict, List, Tuple
import numpy as np
import torch
from overrides import overrides
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM 
# MLM任务专用模型 作用：在BertModel基础上添加MLM预测头，用于预测被遮蔽词。结构：BertModel + 线性分类层（将隐藏状态映射到词汇表）。
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.embedding_strategy.input_embedding_strategy import EmbeddingPreprocessor
# from transformers import BertModel  # BERT的核心架构，输出上下文相关的隐藏状态。包含嵌入层和多层Transformer编码器。输出最后一层（或所有层）的隐藏表示，不直接用于任务预测。
from lexsubgen.embedding_strategy.output_embedding_strategy import outputlogits_stategy
from lexsubgen.utils.lemmatize import lemmatize_candidates
from captum.attr import LayerIntegratedGradients
from lexsubgen.findOtherPosition.findOtherPositions import findOtherPositions



class BertProbEstimator(EmbSimProbEstimator):
    # 类属性，用于存储加载的词嵌入和分词器
    _word_embeddings = None
    _tokenizer = None

    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "bert-large-cased",        # 不同于config中的bert
        embedding_similarity: bool = False, 
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        sim_func: str = "dot-product",
        use_subword_mean: bool = False,         # 未出现的词取平均，就调用父类方法
        verbose: bool = False,
        cuda_device: int = 0,
    ):
        super(BertProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
            weights=weights,
            stratagy_input_embedding=stratagy_input_embedding,
            generation_otherPosition_ways=generation_otherPosition_ways,
            generation_otherPosition_ways_mode=generation_otherPosition_ways_mode
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.use_subword_mean = use_subword_mean
        self.cuda_device=cuda_device
        
        self.stratagy_input_embedding=stratagy_input_embedding
        self.generation_otherPosition_ways=generation_otherPosition_ways
        self.generation_otherPosition_ways_mode=generation_otherPosition_ways_mode
        self.mixup_alpha = mixup_alpha
        self.synonym_topn = synonym_topn
        self.gauss_sigma = gauss_sigma
        self.weights=weights        # 输出的合并方式
        self.decayrate=decay_rate   #指数方式


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)   # 设置！
        if self.cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "bert",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "use_subword_mean": self.use_subword_mean,
                "target_output_embedding_type":self.weights
            }
        }
        # 初始化预处理器，gauss keep mask mix-up drop-out
        # 外部加载模型并提取词向量
        # from transformers import BertModel
        # bert_model = BertModel.from_pretrained("bert-base-uncased")
        # word_embeddings = bert_model.embeddings.word_embeddings

        self.register_model()   # 默认在cpu     提前加载，放在前面
        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

        # 检查类属性是否已经加载了词嵌入和分词器
        if BertProbEstimator._word_embeddings is None or BertProbEstimator._tokenizer is None:
            BertProbEstimator._word_embeddings = torch.tensor(
                self.loaded[self.model_name]["embeddings"], device=self.device
            )
            BertProbEstimator._tokenizer = self.loaded[self.model_name]["tokenizer"]
        self.preprocessor = EmbeddingPreprocessor(
            word_embeddings=self._word_embeddings,
            # model_name=model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            strategy=stratagy_input_embedding,

            mixup_alpha=mixup_alpha,
            synonym_topn=synonym_topn,
            gauss_sigma=gauss_sigma,
        )

    @property
    def tokenizer(self):
        """
        Model tokenizer.
        Returns:
            `transformers.BertTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]
    

    # 注册模型。如果指定的 BERT 模型尚未加载，则加载模型、分词器、词汇表和词嵌入，并将其注册到类的静态变量 loaded 中。如果模型已加载，则增加其引用计数。这有助于节省计算资源，避免重复加载相同的模型。
    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in BertProbEstimator.loaded: # 父类的loaded
            self.model_name = '/home/ustc_zyhu/LexSub/transformer_models/'+self.model_name

            bert_model = BertForMaskedLM.from_pretrained(self.model_name)       # 实现了模型加载，字典
            bert_model=bert_model.to(self.device).eval()                        # 指定设备
            bert_tokenizer = BertTokenizer.from_pretrained(
                self.model_name, do_lower_case=self.model_name.endswith("uncased")
            )
            bert_word2id = BertProbEstimator.load_word2id(bert_tokenizer)
            bert_filter_word_ids = BertProbEstimator.load_filter_word_ids(
                bert_word2id, punctuation
            )
            word_embeddings = (
                # bert_model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
                # 不训练但还要继续 forward → detach() 就够了，别转 CPU
                # 只做分析，不再给模型用 → detach().cpu().numpy()
                bert_model.bert.embeddings.word_embeddings.weight.detach()
            )
            # 这里将模型的loaded中传入了，因此父类中就可以直接调用了
            BertProbEstimator.loaded[self.model_name] = {
                "model": bert_model,
                "tokenizer": bert_tokenizer,
                "embeddings": word_embeddings,
                "word2id": bert_word2id,
                "filter_word_ids": bert_filter_word_ids,
            }
            BertProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            BertProbEstimator.loaded[self.model_name]["ref_count"] += 1
    # 获取未知词的向量表示。如果 use_subword_mean 为 True，则将未知词拆分为子词，并返回这些子词嵌入的平均值。
    # 否则，调用父类的 get_unk_word_vector 方法获取默认的零向量。
    # bert词汇表可能不包含这个单词
    @overrides
    def get_unk_word_vector(self, word) -> np.ndarray:
        """
        This method returns vector to be used as a default if
        word is not present in the vocabulary. If `self.use_subword_mean` is true
        then the word will be splitted into subwords and mean of their embeddings
        will be taken.
        Args:
            word: word for which the vector should be given

        Returns:
            zeros vector
        """
        if self.use_subword_mean:
            sub_token_ids = self.tokenizer.encode(word)[1:-1]       # CLS esp标记
            # 使用 BERT 分词器（BertTokenizer）将输入的单词 word 编码成模型可以理解的 token ID（整数序列）
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True)
            return mean_vector
        return super(BertProbEstimator, self).get_unk_word_vector(word)
    

        # 加载模型的词汇表，并返回一个从词汇到索引的映射字典。这对于将词汇转换为模型输入的 ID 非常有用
        # pad->0 unk->1    playing->play+##ing代表两个子词
        # 字典的具体内容取决于所使用的 BERT 模型和其训练时使用的词汇表。
        # 不同的模型可能会有不同的词汇表和相应的索引映射。
        # 因此，word2id 字典的内容是模型特定的。
    @staticmethod
    def load_word2id(tokenizer: BertTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.
        Args:
            tokenizer: `transformers.BertTokenizer` tokenizer
        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id
    

    # 根据给定的过滤字符（如标点符号）生成一个词汇表索引的列表，表示需要从输出分布中过滤掉的词汇。这有助于避免模型生成无意义的标点符号。
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


    # 返回需要从输出分布中过滤掉的词汇的索引列表。这些词汇通常是标点符号或其他不需要的词汇。
    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.
        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["filter_word_ids"]


    # 将给定的词汇列表转换为 BERT 模型可以处理的子词列表。它使用 BERT 分词器将每个词汇拆分为子词，并返回这些子词的列表。
    # 非首词会添加“##”，无多余空格，没有字符串拼接          直接作用于词元，所以像 ','两边不存在都有空格的情况  .join(' ')
    def bert_tokenize_sentence(
        self, tokens: List[str], tokenizer: BertTokenizer = None
    ) -> List[str]:
        """
        Auxiliary function that tokenize given context into subwords.

        Args:
            tokens: list of unsplitted tokens.
            tokenizer: tokenizer to be used for words tokenization into subwords.

        Returns:
            list of newly acquired tokens
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        bert_tokens = list()
        for token in tokens:
            bert_tokens.extend(tokenizer.tokenize(token))       # 将输入的文本分割成一个个的词元（tokens）
        return bert_tokens

    # 将一批上下文和目标词索引转换为适合 BERT 模型处理的格式
    # 返回target位置，注意：bert是加CLS和sep，其它模型不一定，在输入模型计算分数的时候，需要添加标记
    def bert_prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: BertTokenizer = None,
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.
        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            tokenizer: tokenizer to use for word tokenization
        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:   # 配置文件有配置，不用默认的
            tokenizer = self.tokenizer

        bert_batch_of_tokens, bert_batch_of_target_ids,bert_len_of_tokens,origin_word = list(), list(),list(),list()
        temp=list()
        max_seq_len = 0     # 最大长度初始为0，并没有规定最大值常量
        # L target R ——sentence句子处理
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            left_context = ["[CLS]"] + self.bert_tokenize_sentence( # 这里是分词后的left_context,target_id变大
                tokens[:target_idx], tokenizer
            )
            right_context = self.bert_tokenize_sentence(
                tokens[target_idx + 1 :], tokenizer
            ) + ["[SEP]"]

            target_tokens = self.bert_tokenize_sentence([tokens[target_idx]], tokenizer)    # 使用bert模型编码成词元， playing = play + ing
            length_target_tokens=1
            # 目标词的mask策略
            if self.mask_type == "masked":
                target_tokens = ["[MASK]"]  # 整体mask
            elif self.mask_type == "not_masked":   
                length_target_tokens=len(target_tokens)
            else:
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")
            
            # xlnet给的灵感，是不是有空格的存在？？？
            context = left_context + target_tokens + right_context
            seq_len = len(context)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            bert_batch_of_tokens.append(context)
            bert_batch_of_target_ids.append(len(left_context))
            bert_len_of_tokens.append(length_target_tokens)
            origin_word.append(tokens[target_idx])        # 注意句首添加了CLS
        

        bert_batch_of_tokens = [
            tokens + ["[PAD]"] * (max_seq_len - len(tokens))
            for tokens in bert_batch_of_tokens
        ]
        return bert_batch_of_tokens, bert_batch_of_target_ids,bert_len_of_tokens,origin_word
    
        
    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
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
            logits = self.predict(tokens_lists, target_ids,target_pos_tag)
            # 下为输出的探究,当考虑多个位置的时候需要
            processed=[]
            for idx,logit in enumerate(logits):
                    # logit 形状为 [num_subwords, vocabsize]
                sample_tensor = torch.tensor(logit)
                aggregated = outputlogits_stategy(sample_tensor.unsqueeze(0), self.weights, self.decayrate)  # 增加一个维度以符合函数输入要求
                processed.append(aggregated.squeeze(0).cpu().numpy())  # 去除多余维度
            logits=processed
            logits=np.vstack(logits)     # （batch_size,vocab_size） 最后经过输出都是1条向量
        logits[:, self.filter_word_ids] = -1e9
        return logits, self.word2id

    def predictByTarget(self,  tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]):
        # 准备batch数据，bert_tokens有子词标记，填充好了的，pad，一个批次的数据相同
        bert_tokens, bert_target_ids, length_target_tokens,original_words = self.bert_prepare_batch(
            tokens_lists, target_ids
        )
        # 转换为input_ids
        input_ids = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(tokens)  for tokens in bert_tokens
        ]).to(self.device)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            attention_mask = attention_mask.to(input_ids)
        target_logits=[]
        # 获取原始嵌入
        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_ids)  # 原始嵌入
            # 遍历处理每个样本
            for idx, (target_pos, subword_len, orig_word) in enumerate(zip(bert_target_ids, length_target_tokens, original_words)):
                target_subwords = bert_tokens[idx][target_pos:target_pos+subword_len]        
                # 直接使用原始词进行预处理
                processed_embeds = self.preprocessor.process_word(
                    original_word=orig_word,  # 直接传入原始词
                    subwords=target_subwords,  # 子词序列，面临分词的风险
                    pos_tag=target_pos_tag[idx][0]
                )
                
                # processed_embeds_tensor = torch.stack(processed_embeds)
                embeddings[idx][target_pos]=processed_embeds

            # 前向传播优化
            outputs = self.model(inputs_embeds=embeddings,attention_mask=attention_mask)
            logits = outputs.logits
            
            # 7. 使用 batch 索引同时提取每个样本在其 target 位置的 logits——就一个位置提取，不对啊,补充长度
            # 只返回的目标位置，默认取第一个
            # batch_indices = torch.arange(logits.size(0)).to(self.device)
            target_positions = torch.tensor(bert_target_ids).to(self.device)
            target_length=torch.tensor(length_target_tokens).to(self.device)
            # target_logits = logits[batch_indices, target_positions,:] / self.temperature

            # 访问多个
            batch_size = logits.size(0)
            target_logits_list = [
                logits[i, target_positions[i] : target_positions[i] + target_length[i], :]/self.temperature
                for i in range(batch_size)
            ]
            # .cpu().numpy()
        return target_logits_list
    

    
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
        # self.model.config.num_hidden_layers  # 24
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

            lig = LayerIntegratedGradients(forward_func, self.model.bert.embeddings)

            baseline_ids = torch.full_like(input_ids_batch, self.tokenizer.pad_token_id).to(self.device)
            baseline_embeds = self.model.bert.embeddings(baseline_ids)
            input_embeds_batch = self.model.bert.embeddings(input_ids_batch)

            attributions, _ = lig.attribute(inputs=input_embeds_batch,
                                            baselines=baseline_embeds,
                                            return_convergence_delta=True)
            
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
                orig_target_vec = orig_hid[start_pos:end_pos].mean(dim=0)
                pert_target_vec = pert_hid[start_pos:start_pos + new_len].mean(dim=0)
                target_sim = F.cosine_similarity(orig_target_vec, pert_target_vec, dim=0).item()
                sims.append(target_sim)

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



    # 实验，候选词生成：mask dropout IG attention
    # 单条数据运行5词，每次取10条数据，将logits stack，得到的最终候选词 为 50批次*（50+1）=2500行的候选词，再拆分——不改变原有代码
    # 加一方便对比原始数据的生成质量
    def predictByOther(self,  tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]):
        # positions:[list[topk=5]] 经过模型分词的索引 （batch_size,top-k）
        
        positions=findOtherPositions(self,tokens_lists, target_ids, self.generation_otherPosition_ways)    # 先找出重要位置

        # 准备batch数据，bert_tokens有子词标记，填充好了的，pad，一个批次的数据相同
        bert_tokens, bert_target_ids, length_target_tokens,original_words = self.bert_prepare_batch(
            tokens_lists, target_ids
        )
        # 转换为input_ids
        input_ids = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(tokens)  for tokens in bert_tokens
        ]).to(self.device)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            attention_mask = attention_mask.to(input_ids)
        target_logits=[]
        # 获取原始嵌入
        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_ids)  # 原始嵌入 [batch_size, seq_len, hidden]
            # 遍历处理每个样本
            batch_size, seq_len, hidden = embeddings.size()
            topk = len(positions[0])

            # 扩展 batch：每个样本生成 topk+1 条（1 原始 + topk 变换）
            all_input_ids = []
            all_attention_mask = []
            all_embeddings = []

            for idx in range(batch_size):
                # 原始样本
                all_input_ids.append(input_ids[idx])
                all_attention_mask.append(attention_mask[idx])
                all_embeddings.append(embeddings[idx])

                # topk 修改样本
                for pos in positions[idx]:
                    if self.generation_otherPosition_ways_mode == "mask":
                        # 替换为 [MASK]
                        new_input_ids = input_ids[idx].clone()
                        new_input_ids[pos] = self.tokenizer.mask_token_id
                        
                        all_input_ids.append(new_input_ids)
                        all_attention_mask.append(attention_mask[idx])
                        new_input_ids = new_input_ids.unsqueeze(0)  # 变成 [1, seq_len]
                        new_embeddings = self.model.bert.embeddings(new_input_ids)
                        all_embeddings.append(new_embeddings.squeeze(0))  # 去掉批次维度

                    else:
                        # 对应 embedding dropout
                        new_embeds = embeddings[idx].clone()
                        # 取出原始 token embedding
                        emb = new_embeds[pos]
                        mask = (torch.rand_like(emb) > 0.5).float()      
                        dropped_embedding = emb * mask
                        new_embeds[pos] = dropped_embedding

                        all_input_ids.append(input_ids[idx])
                        all_attention_mask.append(attention_mask[idx])
                        all_embeddings.append(new_embeds)

            # 拼接
            all_input_ids = torch.stack(all_input_ids).to(self.device)               # [batch_size*(topk+1), seq_len]
            all_attention_mask = torch.stack(all_attention_mask).to(self.device)     # [batch_size*(topk+1), seq_len]
            all_embeddings = torch.stack(all_embeddings).to(self.device)             # [batch_size*(topk+1), seq_len, hidden]

            # 前向传播
            outputs = self.model(inputs_embeds=all_embeddings, attention_mask=all_attention_mask)
            logits = outputs.logits  # [batch_size*(topk+1), seq_len, vocab]

            # 提取目标位置 logits
            target_positions = torch.tensor(bert_target_ids).to(self.device)
            target_length = torch.tensor(length_target_tokens).to(self.device)

            expanded_logits = []
            for i in range(logits.size(0)):
                orig_idx = i // (topk+1)  
                start = target_positions[orig_idx]
                length = target_length[orig_idx]
                expanded_logits.append(logits[i, start:start+length, :] / self.temperature)

        return expanded_logits
    

    # 两种生成词汇的方式：原词50个 VS 原词+其余位置=50个
    def predict(self,  tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]):
        if self.generation_otherPosition_ways=='target':
            return self.predictByTarget(tokens_lists, target_ids,target_pos_tag)
        else:
            return self.predictByOther(tokens_lists, target_ids,target_pos_tag)
    
    # version 2: average every layer then add up
    #def getTargetPositionConcatScore(self, tokens_lists=None, pred_substitutes=None, target_ids=None):
        """
        Args:
            tokens_lists: list[list[str]], 每句话的 token 序列
            pred_substitutes: list[list[str]], 每句话的候选替换词
            target_ids: list[int], 每句话目标词的索引（词级别，不是subword级别）

        Returns:
            batch_scores: list，每句话对应候选词的相似度列表
        """
        batch_scores = []
        pred_substitutes = lemmatize_candidates(target_ids, tokens_lists, pred_substitutes)

        for tokens, substitutes, target_id in zip(tokens_lists, pred_substitutes, target_ids):
            orig_word = tokens[target_id]

            # === step1: 构造替换后的句子集合（第一句是原句，后面是候选替换句） ===
            sentences = []
            sentences.append(tokens.copy())  # 原句
            for sub in substitutes:
                new_tokens = tokens.copy()
                new_tokens[target_id] = sub
                sentences.append(new_tokens)

            # === step2: tokenizer 编码 ===
            encodings = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                is_split_into_words=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # list[num_layers+1](batch, seq_len, hidden_dim)

            # === step3: 取第3层到倒数第3层 (去掉前2层和最后2层) ===
            concat_reps = []
            # step3: 在 3 到 L-2 层取 hidden states，并对目标词 subwords 平均，再拼接
            concat_reps = []
            for layer_hs in hidden_states[3:-2]:
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

            # === step4: 相似度计算 ===
            layer_sims = []  # 存放每层的相似度结果
            for layer_reps in concat_reps:  # concat_reps 里每个元素是 (batch, hidden_dim)，一层
                orig_vec = layer_reps[0]        # (hidden_dim,)
                subs_vecs = layer_reps[1:]      # (num_cands, hidden_dim)

                sims = [F.cosine_similarity(orig_vec.unsqueeze(0), cv.unsqueeze(0)).item()for cv in subs_vecs]
                layer_sims.append(sims)  # 每层的相似度列表

            # 把所有层的结果转成 tensor，方便取平均
            layer_sims = torch.tensor(layer_sims)  # (num_layers, num_cands)
            avg_sims = layer_sims.mean(dim=0)      # (num_cands,)

            batch_scores.append(avg_sims.tolist())

        return batch_scores
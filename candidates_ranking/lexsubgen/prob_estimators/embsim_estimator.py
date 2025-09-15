'''
一个基于词嵌入相似度的概率估计器 EmbSimProbEstimator，用于根据目标词和替代词的嵌入相似度来获取替代词的分布。
target词和词汇表（bert模型）的相似度计算，返回logits
'''
import gc
import torch
from collections import defaultdict
from typing import List, Dict
import numpy as np
from scipy.spatial.distance import cdist
from torch.cuda import empty_cache
from lexsubgen.prob_estimators import BaseProbEstimator
import torch.nn.functional as F

SIMILARITY_FUNCTIONS = ("dot-product", "cosine", "euclidean")   # 可用的相似度计算函数名称


class EmbSimProbEstimator(BaseProbEstimator):
    loaded = defaultdict(dict)

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        sim_func: str = "dot-product",
        temperature: float = 1.0,
        weights:str=None,
        stratagy_input_embedding:str=None,
        generation_otherPosition_ways:str=None,
        generation_otherPosition_ways_mode:str=None, 
    ):
        """
        Class that provides an ability to acquire substitutes distribution
        according to the embedding similarity of the target word and a substitute.
        Args:
            model_name: name of the underlying vectorization model.
            verbose: verbosity level, if its true would print some misc info.
            sim_func: name of the method to use in order to compute similarity score.
            temperature: temperature that should be applied to the output logits.
        """
        super(EmbSimProbEstimator, self).__init__(verbose=verbose,weights=weights,stratagy_input_embedding=stratagy_input_embedding,
                                                  generation_otherPosition_ways=generation_otherPosition_ways,
                                                  generation_otherPosition_ways_mode=generation_otherPosition_ways_mode)
        self.model_name = model_name
        self.temperature = temperature
        if sim_func not in SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Wrong name of the similarity function. Choose one from {SIMILARITY_FUNCTIONS}."
            )
        self.sim_func = sim_func
    # 该方法用于将模型添加到内存缓存中，但目前尚未实现
    def register_model(self):
        """
        Method that adds model to the memory cache if not already.
        """
        raise NotImplementedError()


    # 会自动加标记[cls][sep]，bert模型是BertForMaskedLM，没有tokenizer,不同组件分别调用
    # 直接使用 padding=True 可能导致错误，因为默认情况下可能无法推断填充长度。padding='max_length'
    # 每一个坑都得踩一遍！不填充的话，默认只在句子后面加sep，这样每个句子的长度不同，无法计算句子的相似度
    # @property————--只读属性
    def tokenize(self, sentence: str):
        tokenizer=self.loaded[self.model_name]["tokenizer"]
        return tokenizer(sentence,return_tensors='pt',padding='max_length', max_length=256,truncation=True)  # 句子长度大于128
    


    # 遍历目标词，若词在词汇表中，则获取其对应的词嵌入；否则，调用 get_unk_word_vector 方法获取未知词向量。(遍历指的是一个批次的遍历)
    # 目标词和词汇表中所有的词的相似度计算值       +emb
    # 同样作者默认没分词——既然没分词，那么怎么在词汇表中找得到呢？   不是经过模型输出得到，而是embedding
    def get_emb_similarity(
        self, tokens_batch: List[List[str]], target_ids_batch: List[int],
    ) -> np.ndarray:
        """
        Computes similarity between target words and substitutes
        according their embedding vectors.
        Args:
            tokens_batch: list of contexts
            target_ids_batch: list of target word ids in the given contexts
        Returns:
            similarity scores between target words and
            words from the model vocabulary.    ——模型字典中的词
            计算目标词和词汇表中的所有词的相似度！！！！！！！！！！

            traget词和模型中字典的词的相似度,在这儿并没有考虑到target位置处的分词,能分词不分，那在模型中就找不到相近的词啊。
        """
        target_words = [
            tokens[target_idx]
            for tokens, target_idx in zip(tokens_batch, target_ids_batch)
        ]

        target_word_embeddings = []
        for word in target_words:
            if word in self.word2id:
                target_word_embeddings.append(self.embeddings[self.word2id[word]])
            else:
                target_word_embeddings.append(self.get_unk_word_vector(word))    
                # 这里存在问题：维度是多少？一个target词有多个emb的可能吗？
        target_word_embeddings = np.vstack(target_word_embeddings)

        if self.sim_func == "dot-product":
            logits = np.matmul(target_word_embeddings, self.embeddings.T)
            # 每行对应词汇表中的一个词，而每列则是该词的向量表示。
        else:
            # 通常，相似度和距离是反相关的：距离越小，相似度越高。距离越大，相似度越低。
            logits = 1 - cdist(
                target_word_embeddings, self.embeddings, self.sim_func
            )
        logits /= self.temperature      # 除以温度系数
        return logits
    
    # 用于获取未知词的向量，默认返回一个零向量。需要重写该方法以实现自定义逻辑。找不到词也可能是还没分词的影响
    # 所以，同输入处理，用mix-up技术，用0向量去和词汇表中的词做点积，找到的只能是0
    # 分词后再找，找得到就取平均，找不到就返回高斯噪声

    # def get_unk_word_vector(self, word: str) -> np.ndarray:
    #     """
    #     This method returns vector to be used as a default if
    #     word is not present in the vocabulary. You may override
    #     this method in order to implement custom logic.
    #     Args:
    #         word: word for which the vector should be given
    #     Returns:
    #         zeros vector
    #     """
    #     # raise NotImplementedError("Override this method")
    #     embedding_dim = self.embeddings.shape[1]
    #     zeros_vector = np.zeros((1, embedding_dim))
    #     return zeros_vector                  
    
    # zero是不是效果还更好？复现不出论文3
    def get_unk_word_vector(self,word:str)->np.ndarray:
        subwords = self.tokenizer.tokenize(word)
        subword_embeddings = []
        embedding_dim = self.embeddings.shape[1]

        for subword in subwords:
            if subword in self.word2id:
                # 如果子词在词汇表中，获取对应的词嵌入
                index = self.word2id[subword]
                subword_embedding=self.embeddings[index]
            else:
                # 如果子词不在词汇表中，添加高斯噪声---0向量
                # subword_embedding = np.random.normal(0, 1, embedding_dim)
                subword_embedding = np.zeros((1, embedding_dim))
            subword_embeddings.append(subword_embedding)

        # 将子词的表示取平均，也就成了一个维度了。总不能这里也考虑多维吧？那怎么找词？每一维度都找吗？必要性不大，平均了已经
        if subword_embeddings:
            word_embedding = np.mean(subword_embeddings, axis=0)
            return word_embedding
        else:
            subword_embedding = np.zeros((1, embedding_dim))
            return subword_embedding

    # 返回模型的词汇表，即词到索引的映射字典。
    @property
    def word2id(self) -> Dict[str, int]:
        """
        Attribute that acquires model vocabulary.

        Returns:
            vocabulary represented as a `dict`
        """
        return self.loaded[self.model_name]["word2id"]
    # 返回模型的词嵌入矩阵。通过word2id索引得到
    @property
    def embeddings(self) -> np.ndarray:
        """
        Attribute that acquires model word embeddings.

        Returns:
            2-D `numpy.ndarray` with rows representing word vectors.
        """
        return self.loaded[self.model_name]["embeddings"]
    # 返回底层的向量模型。
    @property
    def model(self):
        """
        Attribute that acquires underlying vectorization model.

        Returns:
            Vectorization model.
        """
        return self.loaded[self.model_name]["model"]


    # 此处定义，子类都能实现调用    zip组合成元组，用元组进行迭代
    # hidden_state:(batch_size, sequence_length, hidden_size) flatten——(batch_size, sequence_length * hidden_size)
    # 需要排除掉padding处的影响
    # 忽略了CLS标记的影响，严格而言，提取logits需要target_id+1
    # 这里是评估，和具体的模型的选词分开
    def get_model_score(self, sentences: List[List[str]] = None, pred_substitutes: List[List[str]] = None, target_ids: List[List[int]] = None):
        ans_cls_similarity = []
        ans_token_similarity_score = []
        ans_attention_scores = []
        ans_validation_score=[]
        # 一条句子多个替代词
        for sentence, substitute, target_id in zip(sentences, pred_substitutes, target_ids):
            attention_scores,token_similarity_scores,cls_similarities,validation_score=[],[],[],[]
            # 复制 token 列表，准备替换指定位置的 token
            original_sentence_tokens = sentence.copy()
            # 拼接 token 列表为字符串，便于 tokenizer 正确处理
            original_sentence_str = " ".join(original_sentence_tokens)
            original_inputs = self.tokenize(sentence=original_sentence_str)
            orig_mask = original_inputs['attention_mask']
            combined_mask = orig_mask.float()
            original_outputs = self.model(**original_inputs, output_hidden_states=True, output_attentions=True)
            # 计算 CLS 向量的余弦相似度，LM没有last_hidden_state
            original_cls = original_outputs.hidden_states[-1][:, 0, :]  # shape: (1, hidden_dim)
            # token 级别的 hidden states 比较（只比较有效 token，取顶四层）
            original_top_four_hidden_states = original_outputs.hidden_states[-4:]
            # 最后四层的token表示拼接
            orginal_validation=torch.cat(
                    original_outputs.hidden_states[-4:],  # 取最后四层
                    dim=-1  # 在隐藏层维度拼接，得到拼接后的上下文表示
                )

            new_sentence_tokens = sentence.copy()
            for word in substitute:
                new_sentence_tokens[target_id] = word
                new_sentence_str = " ".join(new_sentence_tokens)
                new_inputs = self.tokenize(sentence=new_sentence_str)
                
                # 新句和原始句的长度不同导致mask中的0或者1数量不同，取最长的还是最短的
                # 获取 attention mask，假设两个输入的 mask 形状一致
                # attention_mask 的 shape: (1, seq_len)
                new_mask = new_inputs['attention_mask']
                
                # 或，可以分别考量，得到的atten值居然有2.几的
                # combined_mask= (orig_mask | new_mask).float()
                # 与
                if orig_mask.shape!=new_mask.shape:
                    orig_mask=orig_mask[:,:256]
                    new_mask=new_mask[:,:256]
                    print(sentence)
                combined_mask = (orig_mask & new_mask).float()

                # 这里假设两者完全一致，实际中可根据需求调整为二者的交集或其它处理
                new_outputs = self.model(**new_inputs, output_hidden_states=True, output_attentions=True)
                new_cls = new_outputs.hidden_states[-1][:, 0, :]
                
                # 新句子的顶四层token表示
                new_validation=torch.cat(
                    new_outputs.hidden_states[-4:],  # 取最后四层，bert-base:768，每个token维度
                    dim=-1  # 在隐藏层维度拼接，得到拼接后的上下文表示  1024*4=4096，维度（1，128，4096）
                )
                
                cls_similarity = F.cosine_similarity(original_cls, new_cls, dim=-1).item()
                cls_similarities.append(cls_similarity)

            
                new_top_four_hidden_states = new_outputs.hidden_states[-4:]
                val_score = 0.0
                for orig_hidden, new_hidden in zip(original_top_four_hidden_states, new_top_four_hidden_states):
                    # orig_hidden, new_hidden shape: (1, seq_len, hidden_dim)，替换后直接导致tokens不同！（会分词）
                    # 计算每个 token 的余弦相似度，结果 shape: (1, seq_len)
                    token_sim = F.cosine_similarity(orig_hidden, new_hidden, dim=-1)
                    # 使用 combined_mask 仅保留有效 token 的相似度
                    # 注意：combined_mask 的 shape: (1, seq_len)
                    valid_count = combined_mask.sum()
                    if valid_count > 0:
                        layer_sim = (token_sim * combined_mask).sum() / valid_count
                    else:
                        layer_sim = token_sim.mean()
                    val_score += layer_sim.item()
                val_score /= 4
                token_similarity_scores.append(val_score)

                # token 级别的 attention 比较（只比较有效 token 的注意力分布，取顶四层）注意力分数而不是注意力权重（权重固定）
                original_top_four_attentions = original_outputs.attentions[-4:]
                new_top_four_attentions = new_outputs.attentions[-4:]
                
                attn_score = 0.0
                total_heads=0
                # 初始化总注意力分数和总头数，[1, 16, 128, 128]，16个头，total_attn应该是二维
                total_attn = torch.zeros(original_outputs.hidden_states[-1].shape[1],original_outputs.hidden_states[-1].shape[1])   # 128
                for orig_attn, new_attn in zip(original_top_four_attentions, new_top_four_attentions):
                    # orig_attn, new_attn shape: (1, num_heads, seq_len, seq_len)
                    # 第四维（L）：代表键（key）的位置，即每个 token 作为键时被其他 token 关注的程度。
                    # 先对每个 head 的每个 token（query）的注意力分布计算余弦相似度（针对 key 维度）
                    # 得到 shape: (1, num_heads, seq_len)=（1，16，128）
                    attn_sim = F.cosine_similarity(orig_attn, new_attn, dim=-1)
                    # 对 query 位置进行 mask（padding 的 token 置零，不参与计算）
                    # mask shape: (1, 1, seq_len)
                    mask_for_attn = combined_mask.unsqueeze(1)
                    mask_for_attn_expand=mask_for_attn.expand(attn_sim.shape)
                    valid_count_attn = mask_for_attn_expand.sum()
                    if valid_count_attn > 0:    # 扩展是逻辑上的虚拟操作（1，16，128）*（1，1，128）相当于值复制了
                        layer_attn = (attn_sim * mask_for_attn).sum() / valid_count_attn
                    else:
                        layer_attn = attn_sim.mean()
                    attn_score += layer_attn.item()

                    # validation中的W
                    # 移除批次维度（假设批次为1）
                    new_attn = new_attn.squeeze(0)  # 形状变为 [头数, token数, token数]=[16,128,128]即16头
                    # 遍历每个头
                    for head_attn in new_attn:
                        # 累加
                        total_attn += head_attn
                        total_heads += 1

                attn_score /= 4
                attention_scores.append(attn_score)

                # 计算平均注意力权重,维度：[seq_length]
                w = total_attn / total_heads
                org_new_sim=F.cosine_similarity(orginal_validation,new_validation,dim=-1)

                mid_validation_score=torch.sum(torch.matmul(org_new_sim,w))
                validation_score.append(mid_validation_score.detach().cpu().numpy().item())  # 矩阵乘法

            ans_attention_scores.append(attention_scores)
            ans_cls_similarity.append(cls_similarities)
            ans_token_similarity_score.append(token_similarity_scores)
            ans_validation_score.append(validation_score)
        return ans_token_similarity_score, ans_cls_similarity, ans_attention_scores,ans_validation_score
    
    
    # def tokenize_batch(self,batch_sentences: List[str]):
    #     # 自动添加特殊标记（[CLS], [SEP]）、填充（padding）和截断（truncation）
    #     return self.tokenize(
    #         batch_sentences,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt"  # 返回PyTorch张量
    #     )
    # def extract_batch_output(batch_outputs, index: int):
    #     # 假设 batch_outputs 是模型返回的包含 hidden_states 和 attentions 的对象
    #     return {
    #         "hidden_states": [layer[index] for layer in batch_outputs.hidden_states],
    #         "attentions": [layer[index] for layer in batch_outputs.attentions],
    #         "cls": batch_outputs.last_hidden_state[index, 0, :]
    #     }




    

        
        
import sys
import os
import argparse

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设项目根目录在上级的上级目录）
project_root = os.path.dirname(os.path.dirname(current_dir))
# 将根目录添加到sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(f'当前目录是：{current_dir}，项目根目录是：{project_root}')

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import numpy as np
import pandas as pd
# from fire import Fire
from overrides import overrides
from collections import OrderedDict
from tqdm import tqdm

from lexsubgen.data.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import (
    compute_precision_recall_f1_topk,
    compute_precision_recall_f1_vocab,
    compute_oot_best_metrics,
    get_mode
)
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
from lexsubgen.metrics.swords_fa_fc_metrics import evaluate_single_instance_fa_fc
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import build_from_config_path,read_config
from lexsubgen.utils.wordnet_relation import to_wordnet_pos, get_wordnet_relation
from lexsubgen.prob_estimators.electra_estimator import ElectraProbEstimator    # replaced token detectino score
from sentence_transformers import SentenceTransformer, util
from lexsubgen.candidates_from_wordnet.from_wordnet import created_proposed_list
from lexsubgen.candidates_from_wordnet.wordnet import Wordnet
logger = logging.getLogger(Path(__file__).name)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent / "run"/"debug" / Path(__file__).stem

# print("lexsubgen/evaluations/lexsub.py DEFAULT_RUN_DIR=")
# print(DEFAULT_RUN_DIR)

# 可解释性梯度
def sort_substitutes_by_gratitude_score(
    otherPosition: List[List[Dict]],
    pred_substitutes: List[List[str]],
    mode: str = "first",  # 可选: "first", "mean", "topk_mean"
) -> List[List[str]]:
    """
    Args:
        otherPosition: list[list[dict]], 每条句子中每个候选词对应的相似度信息
        pred_substitutes: list[list[str]], 原始候选词列表
        mode: str, 排序策略
            - "first": 按 sims 的第一个值排序
            - "mean": 按 sims 的平均值排序
            - "topk_mean": 按 sims 前 k 个平均值排序 (默认取3个)
    Returns:
        sorted_substitutes: list[list[str]], 按得分排序后的候选词
    """
    sorted_substitutes = []

    for sent_idx, (subs_list, pred_list) in enumerate(zip(otherPosition, pred_substitutes)):
        scores = []
        for idx,sub_dict in enumerate(subs_list):
            sims = sub_dict["sims"]  # 提取 sim 数值
            if mode == "first": # 1
                score = sims[0] 
            elif mode == "mean":        # 3
                score = sum(sims)
            elif mode == "topk_mean":
                topk = 2    # 2
                top_sims = sims[:topk]
                score = sum(top_sims)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            scores.append(score)

        # 按 score 从高到低排序 pred_list
        subs_sorted = [sub for _, sub in sorted(zip(scores, pred_list), key=lambda x: x[0], reverse=True)]
        sorted_substitutes.append(subs_sorted)

    return sorted_substitutes


class LexSubEvaluation(Task):
    def __init__(
        self,
        substitute_generator: SubstituteGenerator = None,   # 传入的对象实例，模型做词汇替换 {post_processing、pre_processing、prob_estimator}
        dataset_reader: DatasetReader = None,
        verbose: bool = True,
        k_list: List[int] = (1, 3, 10),
        batch_size: int = 50,
        # 三个参数干啥用的
        save_instance_results: bool = True,
        save_wordnet_relations: bool = False,
        save_target_rank: bool = False,
    ):
        """
        Main class for performing Lexical Substitution task evaluation.
        This evaluation computes metrics for two subtasks in Lexical Substitution task:
        - Candidate-ranking task (GAP, GAP_normalized, GAP_vocab_normalized).
        - All-word-ranking task (Precision@k, Recall@k, F1@k for k-best substitutes).
        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read datasets for Lexical Substitution task.
            verbose: Bool flag for verbosity.
            k_list: List of integer numbers for metrics. For example, if 'k_list' equal to [1, 3, 5],
                then there will calculating the following metrics:
                    - Precion@1, Recall@1, F1-score@1
                    - Precion@3, Recall@3, F1-score@3
                    - Precion@5, Recall@5, F1-score@5
            batch_size: Number of samples in batch for substitute generator.
        """
        super(LexSubEvaluation, self).__init__(
            substitute_generator=substitute_generator,
            dataset_reader=dataset_reader,
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.k_list = k_list
        self.save_wordnet_relations = save_wordnet_relations
        self.save_target_rank = save_target_rank
        self.save_instance_results = save_instance_results

        self.gap_metrics = ["gap", "gap_normalized", "gap_vocab_normalized"]
        # self.base_metrics = ["precision", "recall", "f1_score"]
        # self.oot_best_metrics=['oot','ootm','best','bestm']
        # self.fa_fc_metrics = ['strict_Fa', 'strict_Fc', 'lenient_Fa', 'lenient_Fc','lenient_Fc_un','lenient_Fa_un','strict_Fa_un','strict_Fc_un']
        # k_metrics = []
        # for k in self.k_list:
        #     k_metrics.extend([f"prec@{k}", f"rec@{k}", f"f1@{k}"])
        self.metrics = self.gap_metrics + self.base_metrics + k_metrics+self.oot_best_metrics+self.fa_fc_metrics

    @overrides
    def get_metrics(self, dataset: pd.DataFrame,**kwargs) -> Dict[str, Any]:
        """
        子类重写父类方法，参数数量需要保持一致，keyword aruguments，表示可接受多个参数
        Method for calculating metrics for Lexical Substitution task.
        Args:
            dataset: pandas DataFrame with the whole datasets.
        Returns:
            metrics_data: Dictionary with two keys:
                - all_metrics: pandas DataFrame, extended 'datasets' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
        """
        logger.info(f"Lexical Substitution for {len(dataset)} instances.")

        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["gold_subst"].tolist(),
            dataset["gold_subst_weights"].tolist(),
            dataset["candidates"].tolist(),
            dataset["target_lemma"].tolist(),   # “target_lemma” 指的是目标词的词元形式。 cats——cat
            batch_size=self.batch_size, # 批量
        )
        # 添加进度条，批量读取数据
        if self.verbose:
            progress_bar = tqdm(
                progress_bar,
                desc=f"Lexical Substitution for {len(dataset)} instances"
            )

        all_metrics_data, columns = [], None
        
        num_mode=0.0      # 数据集中，有多少数据有mode,方便计算bestm和ootm
        for (
            tokens_lists,
            target_ids,
            pos_tags,
            gold_substitutes,
            gold_weights,
            candidates, 
            target_lemmas,
        ) in progress_bar:
            
            
            # probs, word2id = self.substitute_generator.get_probs(
            #     tokens_lists, target_ids, pos_tags
            # )
            #
            # pred_substitutes,pred_substitutes_and_probs = self.substitute_generator.substitutes_from_probs(
            #     probs, word2id, tokens_lists, target_ids
            # )
            #
            # clean candidates
            for i in range(len(candidates)):
                candidates[i]=[w for w in candidates[i] if " " not in w and "-" not in w]  # 过滤掉短语

            # 0. baseline


            # 1.目标位置
            # targetPositionConcatScore=self.substitute_generator.getTargetPositionConcatScore(tokens_lists,candidates,target_ids)
            # ranked_candidates_by_final_scores=sort_substitutes_by_score(targetPositionConcatScore, candidates)  

            # # 4.weight
            # otherPositionConcatScore=self.substitute_generator.getOtherPositionConcatScoreByOne(tokens_lists,candidates,target_ids)
            # ranked_candidates_by_final_scores=sort_substitutes_by_gratitude_score(otherPositionConcatScore, candidates,"mean")

            # # 3.attention
            otherPositionConcatScore=self.substitute_generator.getOtherPositionConcatScoreByAtten(tokens_lists,candidates,target_ids)
            ranked_candidates_by_final_scores=sort_substitutes_by_gratitude_score(otherPositionConcatScore, candidates,"mean")


            # # 2.ig
            # otherPositionConcatScore=self.substitute_generator.getOtherPositionConcatScoreByIg(tokens_lists,candidates,target_ids)
            # ranked_candidates_by_final_scores=sort_substitutes_by_gratitude_score(otherPositionConcatScore, candidates,"mean")
                        
            ranked = self.substitute_generator.candidates_from_probs(
                probs,word2id,candidates
            )
            ranked_candidates_in_vocab, ranked_candidates = ranked
            
            ranked_candidates_in_vocab=ranked_candidates_by_final_scores  # 测试三种排序方法用
            
            

            # 一次处理一行数据，循环bath次，分数取平均,这里的pred_substitutes维度（batch_size,num_substitutes）
            for i in range(len(pred_substitutes)):
                instance_results = OrderedDict([    # 记住元素插入的顺序
                    ("target_word", tokens_lists[i][target_ids[i]]),
                    ("target_lemma", target_lemmas[i]),
                    ("target_pos_tag", pos_tags[i]),
                    ("target_position", target_ids[i]),
                    ("context", json.dumps(tokens_lists[i])),
                ])

                # # 过滤掉gold中的短语？_________________________________________________________________________________________________
                new_subs = []
                new_weights = []
                for sub, w in zip(gold_substitutes[i], gold_weights[i]):
                    # 判断是否为短语（用空格和-判断）
                    if " " not in sub and "-" not in sub:  
                        new_subs.append(sub)
                        new_weights.append(w)
                gold_substitutes[i] = new_subs
                gold_weights[i] = new_weights
                if len(gold_substitutes[i]) == 0:
                    continue


                # 数据集总共有多少条数据有mode
                mode =get_mode(gold_substitutes[i],gold_weights[i])
                if mode is not None:
                    num_mode+=1


                # Metrics computation
                if "swords" in self.dataset_reader.dataset_path.name:
                    results =evaluate_single_instance_fa_fc(
                        gold_substitutes[i],gold_weights[i], pred_substitutes[i], k=10
                    )
                    # instance_results['strict_Fa'] = results["strict_Fa"]
                    # instance_results['strict_Fc'] = results["strict_Fc"]
                    # instance_results['lenient_Fa'] = results["lenient_Fa"]
                    # instance_results['lenient_Fc'] = results["lenient_Fc"]
                    # instance_results['lenient_Fc_un'] = results["lenient_Fc_un"]
                    # instance_results['lenient_Fa_un'] = results["lenient_Fa_un"]
                    # instance_results['strict_Fa_un'] = results["strict_Fa_un"]
                    # instance_results['strict_Fc_un'] = results["strict_Fc_un"]

                    # 对于swords，计算 GAP 之前，先过滤掉权重为0的词
                    filtered_subs = []
                    filtered_weights = []
                    for sub, w in zip(gold_substitutes[i], gold_weights[i]):
                        if w != 0:      # 此处即后面，只过滤了大于0的，用于计算best oot ootm等：设置不一样，需要说明
                            filtered_subs.append(sub)
                            filtered_weights.append(w)
                    gold_substitutes[i] = filtered_subs
                    gold_weights[i] = filtered_weights
                    if len(gold_substitutes[i]) == 0:
                        continue

                    gap_scores = gap_score(
                    gold_substitutes[i], gold_weights[i],
                    ranked_candidates_in_vocab[i], word2id,
                    )
                    for metric, gap in zip(self.gap_metrics, gap_scores):
                        instance_results[metric] = gap

                    # # Computing basic Precision, Recall, F-score metrics——此指标舍去
                    # base_metrics_values = compute_precision_recall_f1_vocab(
                    #     gold_substitutes[i], word2id
                    # )
                    # for metric, value in zip(self.base_metrics, base_metrics_values):
                    #     instance_results[metric] = value
                    #
                    # # Computing Top K metrics for each K in the k_list
                    # k_metrics = compute_precision_recall_f1_topk(
                    #     gold_substitutes[i], pred_substitutes[i], self.k_list
                    # )
                    # for metric, value in k_metrics.items():
                    #     instance_results[metric] = value
                    #
                    #  # computing oot ootm best bestm，注意是考察的前10个，单条数据还没取平均呢,之前的计算ootm和bestm有失误
                    # oot_and_best=compute_oot_best_metrics(gold_substitutes[i],gold_weights[i],pred_substitutes[i][:10])
                    # for metric,value in oot_and_best.items():
                    #     instance_results[metric]=value

                else:
                    # instance_results['strict_Fa'] = 0.0
                    # instance_results['strict_Fc'] = 0.0
                    # instance_results['lenient_Fa'] = 0.0
                    # instance_results['lenient_Fc'] = 0.0
                    # instance_results['lenient_Fc_un'] = 0.0
                    # instance_results['lenient_Fa_un'] = 0.0
                    # instance_results['strict_Fa_un'] = 0.0
                    # instance_results['strict_Fc_un'] = 0.0

                    # Compute GAP, GAP_normalized, GAP_vocab_normalized and ranked candidates
                    gap_scores = gap_score(
                        gold_substitutes[i], gold_weights[i],
                        ranked_candidates_in_vocab[i], word2id,
                    )
                    for metric, gap in zip(self.gap_metrics, gap_scores):
                        instance_results[metric] = gap

                    # Computing basic Precision, Recall, F-score metrics
                    # base_metrics_values = compute_precision_recall_f1_vocab(
                    #     gold_substitutes[i], word2id
                    # )
                    # for metric, value in zip(self.base_metrics, base_metrics_values):
                    #     instance_results[metric] = value
                    #
                    # # Computing Top K metrics for each K in the k_list
                    # k_metrics = compute_precision_recall_f1_topk(
                    #     gold_substitutes[i], pred_substitutes[i], self.k_list
                    # )
                    # for metric, value in k_metrics.items():
                    #     instance_results[metric] = value
                    #
                    # # computing oot ootm best bestm，注意是考察的前10个，单条数据还没取平均呢,之前的计算ootm和bestm有失误
                    # oot_and_best=compute_oot_best_metrics(gold_substitutes[i],gold_weights[i],pred_substitutes[i][:10])
                    # for metric,value in oot_and_best.items():
                    #     instance_results[metric]=value

                if self.save_instance_results:
                    additional_results = self.create_instance_results(
                        tokens_lists[i], target_ids[i], pos_tags[i],
                        probs[i], word2id, gold_weights[i],
                        gold_substitutes[i], pred_substitutes[i],
                        candidates[i],ranked_candidates[i]
                    )
                    instance_results.update(
                        (k, v) for k, v in additional_results.items()
                    )

                all_metrics_data.append(list(instance_results.values()))

                if columns is None:
                    columns = list(instance_results.keys())
        
        all_metrics = pd.DataFrame(all_metrics_data, columns=columns)

        # 此前计算出错，除以的是总的数据条数，改成有mode的数据条数
        mean_metrics = {}
        # for metric in self.metrics:
        #     if metric in ['ootm', 'bestm']:
        #         value = round(all_metrics[metric].sum(skipna=True) / num_mode * 100, 2)
        #     else:
        #         value = round(all_metrics[metric].mean(skipna=True) * 100, 2)
        #     mean_metrics[metric] = value

        return {"mean_metrics": mean_metrics, "instance_metrics": all_metrics}

    def create_instance_results(
        self,
        tokens: List[str], target_id: int, pos_tag: str, probs: np.ndarray,
        word2id: Dict[str, int], gold_weights: Dict[str, int],
        gold_substitutes: List[str], pred_substitutes: List[str],
        candidates: List[str], ranked_candidates: List[str],
    ) -> Dict[str, Any]:
        instance_results = OrderedDict()
        pos_tag = to_wordnet_pos.get(pos_tag, None)
        target = tokens[target_id]
        instance_results["gold_substitutes"] = json.dumps(gold_substitutes)
        instance_results["gold_weights"] = json.dumps(gold_weights)
        # instance_results["pred_substitutes"] = json.dumps(pred_substitutes)
        instance_results["candidates"] = json.dumps(candidates)
        instance_results["ranked_candidates"] = json.dumps(ranked_candidates)

        if hasattr(self.substitute_generator, "prob_estimator"):
            prob_estimator = self.substitute_generator.prob_estimator
            if target in word2id:
                instance_results["target_subtokens"] = 1
            elif hasattr(prob_estimator, "tokenizer"):
                target_subtokens = prob_estimator.tokenizer.tokenize(target)
                instance_results["target_subtokens"] = len(target_subtokens)
            else:
                instance_results["target_subtokens"] = -1

        if self.save_target_rank:
            target_rank = -1
            if target in word2id:
                target_vocab_idx = word2id[target]
                target_rank = np.where(np.argsort(-probs) == target_vocab_idx)[0][0]
            instance_results["target_rank"] = target_rank

        if self.save_wordnet_relations:
            relations = [
                get_wordnet_relation(target, s, pos_tag)
                for s in pred_substitutes
            ]
            instance_results["relations"] = json.dumps(relations)

        return instance_results

    # 写入结果数据
    @overrides
    def dump_metrics(
        self, metrics: Dict[str, Any], run_dir: Optional[Path] = None, log: bool = False
    ):
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics: Dictionary with two keys:
                - all_metrics: pandas DataFrame, extended 'datasets' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        if run_dir is not None:
            with (run_dir / "metrics.json").open("w") as fp:
                json.dump(metrics["mean_metrics"], fp, indent=4)
            with (run_dir / "metrics.json").open("a") as fp:    # 输入
                json.dump(metrics["stratagy_input_embedding"], fp, indent=4)
            with (run_dir / "metrics.json").open("a") as fp:    # 输出
                json.dump(metrics["weight"], fp, indent=4)

            if self.save_instance_results:
                metrics_df: pd.DataFrame = metrics["instance_metrics"]
                metrics_df.to_csv(run_dir / "results.csv", sep=",", index=False)
                metrics_df.to_html(run_dir / "results.html", index=False)
            if log:
                logger.info(f"Evaluation results were saved to '{run_dir.resolve()}'")
        if log:
            logger.info(json.dumps(metrics["mean_metrics"], indent=4))


    # 程序入口——开始处
    # 指明 模型配置+数据集配置+其它参数
    def solve(
        self,
        substgen_config_path: str,
        dataset_config_path: str,
        run_dir: str = DEFAULT_RUN_DIR,     # 是否需要根据模型+数据名动态变化run_dir，涉及到保存配置文件和运行目录
        mode: str = "evaluate",
        force: bool = False,
        auto_create_subdir: bool = True,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration files.
        Builds datasets reader from datasets dataset_config_path and substitute generator from substgen_config_path.
        Args:
            substgen_config_path: path to a configuration file.
            dataset_config_path: path to a datasets configuration file.
            run_dir: path to the directory where to [store experiment data].
            mode: evaluation mode - 'evaluate' or 'hyperparam_search'
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
            MLFlow
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        config = {
            "class_name": "evaluations.lexsub.LexSubEvaluation",
            "substitute_generator": substgen_config_path,       # 配置不再构建对象，直接保存路径
            "dataset_reader": dataset_config_path,
            "verbose": self.verbose,
            # 开始调用前，实例化了对象，这里规定batch大小
            "k_list": self.k_list,
            "batch_size": self.batch_size,
            "save_instance_results": self.save_instance_results,
            "save_wordnet_relations": self.save_wordnet_relations,
            "save_target_rank": self.save_target_rank,
        }
        from lexsubgen.runner import Runner
        runner = Runner(config,run_dir, force, auto_create_subdir)
        
        if mode == "evaluate":
            runner.evaluate(
                config=config,
                experiment_name=experiment_name,
                run_name=run_name
            )
        elif mode == "hyperparam_search":   # 超参数搜索模式
            runner.hyperparam_search(
                config_path=Path(run_dir) / "config.json",
                experiment_name=experiment_name
            )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-16s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 相对路径，运行时候有问题，别的地方的文件函数运行时候，相对出错——改为绝对
    import os
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(f"绝对路径:{current_dir}")
    # 获取上一层目录
    parent_dir = os.path.dirname(current_dir)
    # 获取上上层目录
    grandparent_dir = os.path.dirname(parent_dir)



    # substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/bert.jsonnet")

    # substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/simcse.jsonnet")
    # substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/mpnet.jsonnet")
    # substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/all_mpnet.jsonnet")
    substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/debertav3.jsonnet")

    # dataset_config_path = os.path.join(current_dir, "../../configs/data/dataset_readers/swords_test.jsonnet")
    # dataset_config_path = os.path.join(current_dir, "../../configs/data/dataset_readers/swords_dev.jsonnet")
    dataset_config_path = os.path.join(current_dir, "../../configs/data/dataset_readers/semeval_all.jsonnet")
    

    lexsub_evaluation = LexSubEvaluation()
    lexsub_evaluation.solve(
        # 跳到base_estimator是否是因为这里的配置是bert??
        substgen_config_path=substgen_config_path,
        dataset_config_path=dataset_config_path,
        run_dir=DEFAULT_RUN_DIR,  # 结果保存目录
        mode='evaluate',  # 评估模式
        force=False,  # 是否覆盖已有数据
        auto_create_subdir=True,  # 是否自动创建子目录
        experiment_name="bert-large-semeval",  # 实验名称
        run_name="test"  # 运行名称
    )

    # 构建配置文件的绝对路径，脚本方式运行
    # parser = argparse.ArgumentParser(description='Run lexical substitution evaluation.')
    # parser.add_argument('--substgen-config-path', type=str, required=True, help='Path to substitute generator config file')
    # parser.add_argument('--dataset-config-path', type=str, required=True, help='Path to dataset config file')
    # parser.add_argument('--run-dir', type=str, default=DEFAULT_RUN_DIR, help='Directory to save results')
    # parser.add_argument('--mode', type=str, default="evaluate", help='Evaluation mode')
    # parser.add_argument('--force', action='store_true', help='Overwrite existing data')
    # parser.add_argument('--auto-create-subdir', action='store_true', help='Automatically create subdirectory')
    # parser.add_argument('--experiment-name', type=str, default="lexsub-all-models", help='Experiment name')
    # parser.add_argument('--run-name', type=str, default="semeval_all_bert", help='Run name')
    #
    # args = parser.parse_args()
    #
    # lexsub_evaluation = LexSubEvaluation()
    # # print(f'配置文件路径：{args.substgen_config_path}')
    # substgen_config_path = os.path.join(grandparent_dir, args.substgen_config_path)
    # dataset_config_path = os.path.join(grandparent_dir, args.dataset_config_path)
    #
    # # 调用solve方法进行任务评估
    # lexsub_evaluation.solve(
    #     # 跳到base_estimator是否是因为这里的配置是bert??
    #     substgen_config_path=substgen_config_path,
    #     dataset_config_path=dataset_config_path,
    #     run_dir=DEFAULT_RUN_DIR,  # 结果保存目录
    #     mode=args.mode,  # 评估模式
    #     force=False,  # 是否覆盖已有数据
    #     auto_create_subdir=True,  # 是否自动创建子目录
    #     experiment_name=args.experiment_name,  # 实验名称
    #     run_name=args.run_name  # 运行名称
    # )

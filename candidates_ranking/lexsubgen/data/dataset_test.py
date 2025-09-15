import sys
import os
from lexsub import LexSubDatasetReader
from pathlib import Path


current_script_dir = Path(__file__).parent

data_root_path = {
    # "coinco": "D:/download/naacl2016-master/data",
    # "semeval_all": "D:/download/naacl2016-master/data/",
    "twsi2": "D:/download/naacl2016-master/data/"
}
url = None
with_pos_tag = True


if __name__=='__main__':
    for dataset_name, path in data_root_path.items():
        try:
            reader = LexSubDatasetReader(
                dataset_name=dataset_name,
                data_root_path=path,
                url=url,
                with_pos_tag=with_pos_tag
            )
            dataset = reader.read_dataset()
            csv_file_path = current_script_dir / f"{dataset_name}.csv"
            dataset.to_csv(csv_file_path, index=False)
            print(f"数据集 {dataset_name} 已成功保存到 {csv_file_path}")
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出现错误: {e}")


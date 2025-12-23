import os
from datasets import load_dataset, DatasetDict

class ToxicDataset:
    """
    用于加载 Toxic Comment Classification 数据集的类。
    封装了 Hugging Face datasets 库的加载逻辑。
    """
    def __init__(self, data_dir):
        """
        初始化数据集加载器
        
        Args:
            data_dir (str): 包含 train.csv, val.csv, test.csv 的目录路径
        """
        self.data_dir = data_dir
        self.label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.dataset = None

    def load_data(self):
        """
        加载 CSV 文件为 Hugging Face DatasetDict
        
        Returns:
            DatasetDict: 包含 train, val, test 分割的数据集
        """
        train_files = [os.path.join(self.data_dir, 'train.csv')]
        aug_train_path = os.path.join(self.data_dir, 'train_augmented_multilingual.csv')
        if os.path.exists(aug_train_path):
            print(f"发现增强训练集: {aug_train_path}，将合并到训练集中。")
            train_files.append(aug_train_path)

        data_files = {
            'train': train_files,
            'val': os.path.join(self.data_dir, 'val.csv'),
            'test': os.path.join(self.data_dir, 'test.csv')
        }
        
        # 检查文件是否存在
        for split, paths in data_files.items():
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"找不到文件: {path}")

        print(f"正在从 {self.data_dir} 加载数据...")
        # 使用 huggingface datasets 加载 CSV
        # cache_dir 可以根据需要配置，这里使用默认
        self.dataset = load_dataset('csv', data_files=data_files)


        print("数据加载完成。")
        print(f"Train: {len(self.dataset['train'])} samples")
        print(f"Val: {len(self.dataset['val'])} samples")
        print(f"Val: {len(self.dataset['val'])} samples")
        print(f"Test: {len(self.dataset['test'])} samples")
        
        return self.dataset

    def get_label_cols(self):
        """
        获取标签列名列表
        """
        return self.label_cols

    def get_num_labels(self):
        """
        获取标签数量
        """
        return len(self.label_cols)

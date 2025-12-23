import pandas as pd
import os
import numpy as np

def split_dataset():
    # 设置随机种子以保证可复现性
    np.random.seed(42)

    # 路径配置
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # 源文件路径
    SOURCE_TRAIN_FILE = os.path.join(PROJECT_ROOT, 'data', 'source', 'train.csv')
    SOURCE_TEST_FILE = os.path.join(PROJECT_ROOT, 'data', 'source', 'test.csv')
    SOURCE_TEST_LABELS_FILE = os.path.join(PROJECT_ROOT, 'data', 'source', 'test_labels.csv')

    # 输出文件路径 (直接在 data 目录下)
    TRAIN_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
    VAL_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'val.csv')
    TEST_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'test.csv')

    # 1. 处理训练集和验证集
    if os.path.exists(SOURCE_TRAIN_FILE):
        print(f"正在读取训练源数据: {SOURCE_TRAIN_FILE}")
        try:
            df = pd.read_csv(SOURCE_TRAIN_FILE)
            
            # 打乱数据
            print("正在打乱数据...")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # 计算切分点 (9:1)
            split_index = int(len(df) * 0.9)

            # 切分数据
            train_df = df.iloc[:split_index]
            val_df = df.iloc[split_index:]

            print(f"原始训练数据量: {len(df)}")
            print(f"划分后 - 训练集: {len(train_df)}, 验证集: {len(val_df)}")

            print("正在保存训练集和验证集...")
            train_df.to_csv(TRAIN_OUTPUT, index=False)
            val_df.to_csv(VAL_OUTPUT, index=False)
            print(f"已保存: {TRAIN_OUTPUT}")
            print(f"已保存: {VAL_OUTPUT}")
            
        except Exception as e:
            print(f"处理训练数据时出错: {e}")
    else:
        print(f"错误: 找不到源文件 {SOURCE_TRAIN_FILE}")

    # 2. 处理测试集 (合并 test.csv 和 test_labels.csv)
    if os.path.exists(SOURCE_TEST_FILE) and os.path.exists(SOURCE_TEST_LABELS_FILE):
        print(f"\n正在处理测试集...")
        try:
            print(f"读取: {SOURCE_TEST_FILE}")
            test_df = pd.read_csv(SOURCE_TEST_FILE)
            print(f"读取: {SOURCE_TEST_LABELS_FILE}")
            test_labels_df = pd.read_csv(SOURCE_TEST_LABELS_FILE)

            # 合并数据
            print("正在合并测试数据和标签...")
            merged_test_df = pd.merge(test_df, test_labels_df, on='id')

            # 过滤掉标签为 -1 的数据 (未标记数据)
            print("正在过滤未标记数据 (label = -1)...")
            original_len = len(merged_test_df)
            merged_test_df = merged_test_df[merged_test_df['toxic'] != -1]
            filtered_len = len(merged_test_df)
            
            print(f"测试集原始数量: {original_len}")
            print(f"过滤后有效测试集数量: {filtered_len}")

            print("正在保存测试集...")
            merged_test_df.to_csv(TEST_OUTPUT, index=False)
            print(f"已保存: {TEST_OUTPUT}")

        except Exception as e:
            print(f"处理测试数据时出错: {e}")
    else:
        print(f"警告: 找不到测试源文件 {SOURCE_TEST_FILE} 或 {SOURCE_TEST_LABELS_FILE}")

    print("\n所有任务完成。")

if __name__ == "__main__":
    split_dataset()

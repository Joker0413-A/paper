import json
import re

def remove_punctuation(text):
    # 正则表达式，匹配任何非Unicode单词字符
    return re.sub(r'[^\w\s]', '', text)

def json_to_bio(json_file_path, output_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for item in data:
            text = item['text']
            # 移除文本中的标点
            original_text = text  # 保留原始文本用于对比
            text = remove_punctuation(text)
            entities = item['entities']
            labels = ['O'] * len(text)  # 初始化标签为 'O'

            # 处理实体标签
            adjusted_entities = []
            for entity in entities:
                original_start = entity['start_idx']
                original_end = entity['end_idx']
                # 计算标点符号影响后的新索引
                # 原始文本到实体开始位置的文本段中的标点符号数量
                punct_before_start = len(re.findall(r'[^\w\s]', original_text[:original_start]))
                # 原始文本到实体结束位置的文本段中的标点符号数量
                punct_before_end = len(re.findall(r'[^\w\s]', original_text[:original_end]))
                start = original_start - punct_before_start
                end = original_end - punct_before_end - 1  # 减1是因为end是包含在内的

                if start < len(text) and end < len(text) and start <= end:
                    adjusted_entities.append((start, end, entity['type']))

            # 应用调整后的实体位置标注BIO
            for start, end, entity_type in adjusted_entities:
                labels[start] = f"B-{entity_type}"
                for i in range(start + 1, end + 1):
                    labels[i] = f"I-{entity_type}"

            # 构建BIO格式的数据并写入文件
            for char, label in zip(text, labels):
                out_file.write(f"{char} {label}\n")
            out_file.write("\n")  # 在实例之间添加新行



# 文件路径配置
base_path = './'  # 假设脚本和数据文件在同一目录下
train_json = base_path + 'CMeEE-V2_train_split.json'
dev_json = base_path + 'CMeEE-V2_dev_split.json'
test_json = base_path + 'CMeEE-V2_test_split.json'

# 输出文件路径
train_output = base_path + 'CMeEE-V2_train_split.txt'
dev_output = base_path + 'CMeEE-V2_dev_split.txt'
test_output = base_path + 'CMeEE-V2_test_split.txt'

# 调用函数进行转换
json_to_bio(train_json, train_output)
json_to_bio(dev_json, dev_output)
json_to_bio(test_json, test_output)

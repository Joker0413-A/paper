import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# 加载数据集
dev_data = load_data('CMeEE-V2_dev.json')
train_data = load_data('CMeEE-V2_train.json')

# 合并数据集
combined_data = dev_data + train_data

# 将数据分割为训练、验证和测试集
train, temp = train_test_split(combined_data, test_size=0.30, random_state=42)  # 先分出30%为测试+验证
valid, test = train_test_split(temp, test_size=0.5, random_state=42)  # 将30%分一半为验证和测试

# 保存数据
def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

save_data(train, 'CMeEE-V2_train_split.json')
save_data(valid, 'CMeEE-V2_dev_split.json')
save_data(test, 'CMeEE-V2_test_split.json')

import os
import pickle

# 假设您的主目录路径为 dir_main
dir_main = '/path/to/your/dataset'
dataset = 'roxford5k'

# 构建标签文件的完整路径
gnd_fname = "/Users/liyunxiao/Desktop/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"
# 加载标签数据
with open(gnd_fname, 'rb') as f:
    gnd_data = pickle.load(f)
# 打印或检查标签数据结构
print(gnd_data.keys())

# 打印 'gnd' 键中第一个元素的键
if len(gnd_data['gnd']) > 0:
    print(gnd_data['gnd'][0].keys()) 
    # dict_keys(['bbx', 'easy', 'hard', 'junk'])

else:
    print("gnd 列表为空")

# 打印 'imlist' 和 'qimlist' 的内容长度和一些示例元素
print(f"imlist 长度: {len(gnd_data['imlist'])}")
if len(gnd_data['imlist']) > 0:
    print(f"imlist 示例: {gnd_data['imlist'][:5]}")  # 打印前5个元素

print(f"qimlist 长度: {len(gnd_data['qimlist'])}")
if len(gnd_data['qimlist']) > 0:
    print(f"qimlist 示例: {gnd_data['qimlist'][:5]}")  # 打印前5个元素

gnd_entry = gnd_data['gnd']
# 打印 'gnd' 键中第一个元素中各个键的内容
# gnd_entry = gnd_data['gnd'][1]
print("Bounding Box (bbx):", gnd_entry['bbx'])
print("Easy Matches:", gnd_entry['easy'])
print("Hard Matches:", gnd_entry['hard'])
print("Junk Matches:", gnd_entry['junk'])

print(gnd_data['gnd'].shape)
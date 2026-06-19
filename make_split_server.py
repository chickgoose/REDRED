import random

with open('/home/aicompetition30/Dataset/1.competition_trainset/target.txt') as f:
    lines = [l.strip() for l in f if l.strip()]

random.seed(42)
random.shuffle(lines)

n_val = int(len(lines) * 0.1)
val   = lines[:n_val]
train = lines[n_val:]

with open('/home/aicompetition30/yolov7/data/val.txt', 'w') as f:
    f.write('\n'.join(val))

with open('/home/aicompetition30/yolov7/data/train.txt', 'w') as f:
    f.write('\n'.join(train))

print(f'train={len(train)}, val={len(val)}')

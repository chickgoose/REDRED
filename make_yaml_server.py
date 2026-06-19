import yaml

with open('/home/aicompetition30/Dataset/names.txt') as f:
    names = [l.strip() for l in f if l.strip()]

cfg = {
    'train': '/home/aicompetition30/yolov7/data/train.txt',
    'val': '/home/aicompetition30/yolov7/data/val.txt',
    'test': '/home/aicompetition30/yolov7/data/val.txt',
    'nc': len(names),
    'names': names
}

with open('/home/aicompetition30/yolov7/data/custom.yaml', 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

print(f'Done: nc={len(names)}')
print('First 3:', names[:3])

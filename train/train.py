from ultralytics import YOLO
import torch
import yaml

# 1. config.yaml 파일에서 설정 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. GPU 사용 가능 여부 확인
device = config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 3. 모델 로드
model = YOLO(config['model'])

# 4. 학습 파라미터 준비 (config.yaml에서 로드)
train_params = {
    'data': config['data'],
    'imgsz': config['imgsz'],
    'epochs': config['epochs'],
    'batch': config['batch'],
    'patience': config['patience'],
    'workers': config['workers'],
    'optimizer': config['optimizer'],
    'project': config['project'],
    'name': config['name'],
    'plots': config['plots'],
    'device': device,
    
    # Augmentation parameters
    'hsv_h': config.get('hsv_h', 0.015),
    'hsv_s': config.get('hsv_s', 0.7),
    'hsv_v': config.get('hsv_v', 0.4),
    'degrees': config.get('degrees', 0.0),
    'translate': config.get('translate', 0.1),
    'scale': config.get('scale', 0.5),
    'shear': config.get('shear', 0.0),
    'perspective': config.get('perspective', 0.0),
    'flipud': config.get('flipud', 0.0),
    'fliplr': config.get('fliplr', 0.5),
    'mosaic': config.get('mosaic', 1.0),
    'mixup': config.get('mixup', 0.0),
    'copy_paste': config.get('copy_paste', 0.0),
}

# Advanced parameters (optional)
optional_params = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                   'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl']
for param in optional_params:
    if param in config:
        train_params[param] = config[param]

# 5. 모델 학습
print(f"\nTraining with configuration:")
print(f"  Model: {config['model']}")
print(f"  Epochs: {config['epochs']}, Batch: {config['batch']}")
print(f"  Image size: {config['imgsz']}")
print(f"  Augmentation - HSV: ({config.get('hsv_h')}, {config.get('hsv_s')}, {config.get('hsv_v')})")
print(f"  Save to: {config['project']}/{config['name']}\n")

results = model.train(**train_params)

print(f"\nResults saved to: '{results.save_dir}'")
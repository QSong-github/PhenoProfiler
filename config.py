#commented out variables are handled by argparse in main.py
debug = True
lr = 1e-3
weight_decay = 1e-3 
patience = 2
factor = 0.5

num_classes = 2913   # 2239 for bbbc036, 1540 for bbbc022, 327 for bbbc037, 2913 for all, 用 @ 后
model_name = 'resnet50'

pretrained = True
trainable = True 
temperature = 1.0

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 672
dropout = 0.1

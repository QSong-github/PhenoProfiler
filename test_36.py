import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics

from dataset import PDDDataset
from models import PhenoProfiler
from utils import *

# 配置参数
CONFIG = {
    "model_path": "result/PhenoProfiler/best.pt",
    "save_path": "revision/bbbc036/PhenoProfiler/",
    "data_paths": {
        "image": "/data/boom/bbbc036/images/",
        "embedding": "/data/boom/bbbc036/embedding/",
        "csv": "/data/boom/bbbc036/profiling.csv",
        "moa": "data/bbbc036_MOA_MATCHES_official.csv"
    },
    "reg_param": 1e-2,
    "batch_size": 600,
    "num_features": 672
}

def whitening_transform(meta, features, config):
    """白化处理流程"""
    # 创建well层数据
    wells = aggregate_to_well_level(meta, features, config)
    
    # 获取控制组数据并转换为NumPy数组
    control_mask = wells["Treatment"] == "NA@NA"  # change
    control_features = wells.loc[control_mask, range(config["num_features"])].values
    
    # 初始化白化器
    whitenizer = WhiteningNormalizer(
        control_features,
        reg_param=config["reg_param"]
    )
    
    # 获取所有特征并转换为NumPy数组
    feature_cols = list(range(config["num_features"]))
    features_to_normalize = wells[feature_cols].values
    
    # 执行归一化并转换回DataFrame
    normalized_features = whitenizer.normalize(features_to_normalize)
    wells[feature_cols] = normalized_features
    
    wells.to_csv(os.path.join(config["save_path"], "1_BC_well_level.csv"), index=False)
    return wells


def main():
    model = PhenoProfiler().cuda()
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    
    # Step 1: Get image embeddings
    if not os.path.exists(CONFIG["save_path"] + "PhenoProfiler_alltrain_36test.npy"):
        img_embeddings = get_image_embeddings(
            model=model,
            batch_size=CONFIG["batch_size"],
            data_paths=CONFIG["data_paths"]
        )
        os.makedirs(CONFIG["save_path"], exist_ok=True)
        np.save(CONFIG["save_path"] + "PhenoProfiler_alltrain_36test.npy", img_embeddings.numpy().T)
    
    # Step 2: Process profiles
    features = np.load(CONFIG["save_path"] + "PhenoProfiler_alltrain_36test.npy").T
    
    print("Loaded features shape:", features.shape)
    print("First feature shape:", features[0].shape)
    
    # 确保特征形状正确
    if features[0].shape != (CONFIG["num_features"],):
        features = features.reshape(-1, CONFIG["num_features"])
        print("Reshaped features to:", features.shape)
    
    # 加载和处理元数据
    meta = pd.read_csv(CONFIG["data_paths"]["csv"])
    
    # 白化处理
    wells = whitening_transform(meta, features, CONFIG)
    
    # print("wells:", wells.head())
    # 计算相似度矩阵和MOA匹配
    profiles, Y = prepare_profiles(wells, CONFIG)
    sim_matrix, moa_matches = compute_similarity_matrix(profiles, Y, CONFIG)
    
    # 计算评估指标
    results = calculate_precision_recall(sim_matrix, moa_matches)
    # print(results)
    print_results(results)
    
def print_results(results):
    """打印评估结果"""
    print(f"Folds of Enrichment@1%: {results['enrichment_top1']:.3f}")
    print(f"Mean Average Precision (MAP): {results['map']:.3f}")
    print(f"Recall@1%: {results['recall_top1']:.3f}")
    print(f"Recall@3%: {results['recall_top3']:.3f}")
    print(f"Recall@5%: {results['recall_top5']:.3f}")
    print(f"Recall@10%: {results['recall_top10']:.3f}")

if __name__ == "__main__":
    main()

## Taken from:
## https://gist.github.com/bwhite/3726239
## Note that this code assumes that the ranking lists provided as
## parameters contain all the relevant documents that need to be
## retrived. In other words, it assumes that the list has no misses.
## It's important when using this code to do that, otherwise the
## results will be artificially inflated. If there is need to compute
## recall metrics, other implementations need to be used.
## MAP slides: https://slideplayer.com/slide/2295316/

"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""

import os
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import sklearn.metrics
from tqdm import tqdm
from torch.utils.data import ConcatDataset

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics
from dataset import PDD_Img_Dataset

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

# ========== 数据加载和处理 ==========
def build_loaders_inference(data_paths, batch_size):
    """构建推理数据加载器"""
    dataset = PDD_Img_Dataset(
        image_path=data_paths["image"],
        embedding_path=data_paths["embedding"], 
        CSV_path=data_paths["csv"]
    )
    return DataLoader(
        ConcatDataset([dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

def get_image_embeddings(model, data_paths, batch_size):
    """获取图像嵌入特征"""
    test_loader = build_loaders_inference(data_paths, batch_size)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting features"):
            features = model.image_encoder(batch["image"].cuda())
            embeddings.append(model.image_projection(features))
    return torch.cat(embeddings).cpu().numpy()

class WhiteningNormalizer:
    def __init__(self, controls, reg_param=1e-6):
        # 确保输入是NumPy数组
        controls = np.asarray(controls)
        self.mu = controls.mean(axis=0)
        X = controls - self.mu
        C = (X.T @ X) / X.shape[0]
        s, V = scipy.linalg.eigh(C)
        self.W = V @ np.diag(1./np.sqrt(s + reg_param)) @ V.T

    def normalize(self, X):
        # 确保输入是NumPy数组
        X = np.asarray(X)
        return (X - self.mu) @ self.W

# ========== 评估指标计算 ==========
def moa_matching(profiles, moa_df):
    """生成MOA匹配矩阵"""
    matches = np.zeros((len(profiles), len(profiles)), dtype=bool)
    moa_list = profiles["Metadata_moa.x"].str.replace('|', '___')
    
    for i, ref_moa in enumerate(moa_list):
        for m in ref_moa.split('|'):
            pattern = rf'(^|___){m}($|___)'
            matches[i] |= moa_list.str.contains(pattern).values
    return matches


def aggregate_to_well_level(meta, features, config):
    """聚合到well层级"""
    site_data, site_features = [], []
    
    for plate in tqdm(meta["Metadata_Plate"].unique(), desc="Aggregating sites"):
        plate_meta = meta[meta["Metadata_Plate"] == plate]
        for _, row in plate_meta.iterrows():
            idx = row.name
            if len(features[idx]) == 0:
                continue
                
            # 获取当前样本的特征
            current_feature = features[idx]
            
            # 检查特征形状并处理
            if current_feature.ndim == 1:  # 已经是向量
                feature_vec = current_feature
            else:  # 如果是2D数组(如多个细胞的多个特征)
                feature_vec = current_feature.mean(axis=0)  # 计算细胞间的平均特征
                
            # 确保特征维度正确
            if feature_vec.shape != (config["num_features"],):
                print(f"Warning: Unexpected feature shape at index {idx}: {feature_vec.shape}")
                continue
                
            site_data.append({
                "Plate": plate,
                "Well": row["Metadata_Well"],
                "Treatment": row["Treatment"],
                "Replicate": row["broad_sample_Replicate"],
                "broad_sample": row["Treatment"].split("@")[0]
            })
            site_features.append(feature_vec)
    
    return create_well_df(site_data, site_features, config)


def create_well_df(site_data, site_features, config):
    """创建well层DataFrame"""
    columns1 = ["Plate", "Well", "Treatment", "Replicate"]
    
    # 转换为numpy数组并检查形状
    features_array = np.array(site_features)
    print("Final features array shape:", features_array.shape)  # 调试输出
    
    if features_array.shape[1] != config["num_features"]:
        raise ValueError(
            f"Feature dimension mismatch: expected {config['num_features']}, "
            f"got {features_array.shape[1]}"
        )
    
    wells = pd.DataFrame(site_data)
    feature_df = pd.DataFrame(features_array, columns=range(config["num_features"]))
    return pd.concat([wells, feature_df], axis=1)

def prepare_profiles(wells, config):
    """准备treatment层特征"""
    wells = wells.drop(columns=["Plate", "Well", "broad_sample"])
    profiles = wells.groupby("Treatment").mean().reset_index()
    profiles["broad_sample"] = profiles["Treatment"].str.split("@", expand=True)[0]
    
    # 添加MOA信息
    columns2 = [i for i in range(config["num_features"])] 

    moa_df = pd.read_csv(config["data_paths"]["moa"])
    profiles = pd.merge(profiles, moa_df, left_on="broad_sample", right_on="Var1")
    profiles = profiles[["Treatment", "broad_sample", "Metadata_moa.x"] + columns2].sort_values(by="broad_sample")
    
    return profiles, moa_df

def compute_similarity_matrix(profiles, moa_df, config):
    """计算相似度矩阵和MOA匹配"""
    cos_sim = sklearn.metrics.pairwise.cosine_similarity(
        profiles[range(config["num_features"])]
    )
    return cos_sim, moa_matching(profiles, moa_df)
   
def enrichment_analysis(similarities, moa_matches, percentile=99.):
    threshold = np.percentile(similarities, percentile)

    v11 = np.sum(np.logical_and(similarities > threshold, moa_matches)) 
    v12 = np.sum(np.logical_and(similarities > threshold, np.logical_not(moa_matches)))
    v21 = np.sum(np.logical_and(similarities <= threshold, moa_matches))
    v22 = np.sum(np.logical_and(similarities <= threshold, np.logical_not(moa_matches)))

    V = np.asarray([[v11, v12], [v21, v22]])
    r = scipy.stats.fisher_exact(V, alternative="greater")
    result = {"percentile": percentile, "threshold": threshold, "ods_ratio": r[0], "p-value": r[1]}
    if np.isinf(r[0]):
        result["ods_ratio"] = v22
    return result

def calculate_enrichment(sim_matrix, matches):
    """计算富集分析（返回DataFrame）"""
    enrichment_data = []
    
    for i in range(sim_matrix.shape[0]):
        if matches[i].sum() > 1:  # 仅处理有效查询
            idx = np.arange(sim_matrix.shape[1]) != i  # 排除自身
            analysis = enrichment_analysis(
                sim_matrix[i, idx], 
                matches[i, idx]
            )
            analysis["query_id"] = i
            enrichment_data.append(analysis)
    
    return pd.DataFrame(enrichment_data)


def calculate_precision_recall(sim_matrix, matches):
    """计算精确度和召回率指标"""
    results = {}
    
    # 富集分析（返回单个数值）
    enrichment_df = calculate_enrichment(sim_matrix, matches)
    results['enrichment_top1'] = enrichment_df['ods_ratio'].mean()  # 提取均值
    
    # 平均精度（保持原实现）
    sim_matrix_np = np.asarray(sim_matrix)
    matches_np = np.asarray(matches, dtype=bool)
    aps = []
    valid_queries = 0
    for i in range(sim_matrix_np.shape[0]):
        # 排除自身匹配
        query_sim = sim_matrix_np[i].copy()
        query_matches = matches_np[i].copy()
        query_sim[i] = -1  # 排除自身
        query_matches[i] = False  # 排除自身
        
        # 只计算有正样本的查询
        if query_matches.sum() > 0:
            ap = sklearn.metrics.average_precision_score(query_matches, query_sim)
            aps.append(ap)
            valid_queries += 1
    
    # 使用第三种方法作为默认MAP
    results['map'] = np.mean(aps)
    
    # 不同top比例的召回率（需要修改实现）
    def safe_recall(sim_vector, match_vector, k):
        if match_vector.sum() == 0:
            return np.nan
        top_k = np.argsort(-sim_vector)[1:k+1]  # 排除自身
        return match_vector[top_k].sum() / match_vector.sum()
    
    recall_metrics = {}
    for percent in [1, 3, 5, 10]:
        k = max(int(len(sim_matrix)*percent/100), 1)
        recalls = [
            safe_recall(sim_matrix[i], matches[i], k) 
            for i in range(len(sim_matrix)) 
            if matches[i].sum() > 1  # 仅计算有效查询
        ]
        recall_metrics[f'recall_top{percent}'] = np.nanmean(recalls)
    
    results.update(recall_metrics)
    
    return results

def recall_at_k(sim_matrix, matches, k):
    """计算top-k召回率"""
    recalls = []
    for i in range(len(sim_matrix)):
        if matches[i].sum() > 0:
            top_k = np.argsort(-sim_matrix[i])[1:k+1]
            recalls.append(matches[i, top_k].sum() / matches[i].sum())
    return np.mean(recalls) if recalls else 0

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def interpolated_precision_recall_curve(Y_true_matrix, Y_predict_matrix):
    """ Compute the average precision / recall curve over all queries in a matrix.
    That is, consider each point in the graph as a query and evaluate all nearest neighbors until
    all positives have been found. Y_true_matrix is a binary matrix, Y_predict_matrix is a
    continuous matrix. Each row in the matrices is a query, and for each one PR curve can be computed.
    Since each PR curve has a different number of recall points, the curves are interpolated to 
    cover the max number of recall points. This is standard practice in Information Retrieval research """

    from sklearn.metrics import precision_recall_curve

    # Suppress self matching
    Y_predict_matrix[np.diag_indices(Y_predict_matrix.shape[0])] = -1 # Assuming Pearson correlation as the metric
    Y_true_matrix[np.diag_indices(Y_true_matrix.shape[0])] = False    # Assuming a binary matrix

    # Prepare axes
    recall_axis = np.linspace(0.0, 1.0, num=Y_true_matrix.shape[0])[::-1]
    precision_axis = []

    # Each row in the matrix is one query
    is_query = Y_true_matrix.sum(axis=0) > 1
    for t in range(Y_true_matrix.shape[0]):
        if not is_query[t]: 
            continue
        # Compute precision / recall for each query
        precision_t, recall_t, _ = precision_recall_curve(Y_true_matrix[t,:], Y_predict_matrix[t,:])

        # Interpolate max precision at all recall points
        max_precision = np.maximum.accumulate(precision_t)
        interpolated_precision = np.zeros_like(recall_axis)

        j = 0
        for i in range(recall_axis.shape[0]):
            interpolated_precision[i] = max_precision[j]
            while recall_axis[i] < recall_t[j]:
                j += 1

        # Store interpolated results for query
        precision_axis.append(interpolated_precision[:,np.newaxis])
    
    return recall_axis, np.mean( np.concatenate(precision_axis, axis=1), axis=1)


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def metrics_precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [metrics_precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

# Fraction strong test
def fraction_strong_test(treatment_corr, null, num_treatments):
    null.sort()
    p95 = null[ int( 0.95*len(null) ) ]
    fraction = np.sum([m > p95 for m in treatment_corr])/num_treatments
    print("Treatments tested:", num_treatments)
    print("At 95th percentile of the null")
    print("Fraction strong: {:5.2f}%".format(fraction*100))
    print("Null threshold: {:6.4f}".format(p95))

if __name__ == "__main__":
    import doctest
    doctest.testmod()

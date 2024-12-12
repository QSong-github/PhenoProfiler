import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn
import argparse
import sklearn.metrics

from dataset import PDDDataset
from models import PhenoProfiler
from utils import *

# define hyperparameters
model = PhenoProfiler().cuda()
model_path = "result/PhenoProfiler/last.pt"
save_path = "Fig3/BBBC022/PhenoProfiler_test/"

def build_loaders_inference(batch_size):
    print("Building loaders")
    dataset = PDDDataset(image_path = "../dataset/bbbc022/images/",
               embedding_path = "../dataset/bbbc022/embedding/",
               CSV_path = "../dataset/bbbc022/profiling.csv")
    
    dataset = torch.utils.data.ConcatDataset([dataset])
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    print("Finished building loaders")
    return test_loader

def get_image_embeddings(model_path, model, batch_size):
    test_loader = build_loaders_inference(batch_size)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    print("Finished loading model")
    
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    
    return torch.cat(test_image_embeddings)

img_embeddings = get_image_embeddings(model_path, model, batch_size=600)
features = img_embeddings.cpu().numpy()

if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(save_path + "PhenoProfiler_alltrain_22test" + ".npy", features.T)

# Load metadata of index data
meta = pd.read_csv(os.path.join(f"../dataset/bbbc022/profiling.csv"))
# remove controled wells
# dmso_indices = meta.index[meta['Compound'] == 'DMSO'].tolist()
# meta = meta.drop(dmso_indices)
# meta.reset_index(drop=True, inplace=True)

# features = np.load(save_path+"PhenoProfiler_alltrain_22test.npy").T #  注意 有 T 无 T
# features = np.load(save_path+"DeepProfiler_Alltrain_22test.npy").T #  注意 有 T 无 T


MATRIX_FILE = save_path+"/cos_efn128combinedcellsout_conv6a_1e-2_e30.csv"
REG_PARAM = 1e-2

meta["broad_sample"] = meta["Treatment"].str.split("@", expand=True)[0]


total_single_cells = 0
for i in range(len(features)):
    if len(features[i]) > 0:
        total_single_cells += features[i].shape[0]

num_features = 672
# print(num_features)

site_level_data = []
site_level_features = []

for plate in tqdm(meta["Metadata_Plate"].unique()):
    m1 = meta["Metadata_Plate"] == plate # 给出meta中的全部plate，标记为TRUE。
    # print('m1:', m1)
    wells = meta[m1]["Metadata_Well"].unique()  # 所有的　well 对应的数据
    # print('wells', wells)
    for well in wells:
        # result = meta.query("Metadata_Plate == {} and Metadata_Well == '{}'".format(plate, well))
        # print(type(plate), type(well))  # <class 'numpy.int64'> <class 'str'>
        result = meta.query(f"Metadata_Plate == {plate} and Metadata_Well == '{well}'")  #　plate是数据，不能用’‘
        # result = meta[(meta["Metadata_Plate"] == 20585) & (meta["Metadata_Well"] == 'P24')]
        # print(plate, well, result)
        for i in result.index:
            # print(features[i].shape) # (54, 672) 表示整个site对应的特征，由（128， 128， 5）变化而来。
            if len(features[i]) == 0:
                continue
            mean_profile = features[i]  # 第0轴（即列）计算中位数。
            # mean_profile = np.median(features[i], axis=0)  # 第0轴（即列）计算中位数。
            # print(mean_profile.shape) # (672,) 统一了
            pert_name = result["Treatment"].unique()
            replicate = result["broad_sample_Replicate"].unique()
            if len(pert_name) > 1:
                print(pert_name)
            site_level_data.append(
                {
                    "Plate": plate,
                    "Well": well,
                    "Treatment": pert_name[0],
                    "Replicate": replicate[0],
                    "broad_sample": pert_name[0].split("@")[0]
                }
            )
            site_level_features.append(mean_profile)

columns1 = ["Plate", "Well", "Treatment", "Replicate"]
columns2 = [i for i in range(num_features)]  #　(672)

sites1 = pd.DataFrame(columns=columns1, data=site_level_data)
sites2 = pd.DataFrame(columns=columns2, data=site_level_features)
sites = pd.concat([sites1, sites2], axis=1)

tmp = meta.groupby(["Metadata_Plate", "Metadata_Well", "Treatment", "broad_sample"])["DNA"].count().reset_index() # 对 meta 数据框进行分组，并计算每个分组中 DNA 列的计数
# print('tmp\n', tmp)  #[7407 rows x 5 columns]

#对 sites 数据按 Plate、Well 和 Treatment 列进行分组（保证他们4列没有重复的行），并计算每个分组中（除开这３列的其他所有列）数值列的平均值。
# 结果就是把 well 合并了。因为 well 和 Treatment 是对应的。
wells = sites.groupby(["Plate", "Well", "Treatment", "Replicate"]).mean().reset_index()
# print(wells.shape)  # (7407, 676)  7407 个 wells
# wells

wells = pd.merge(wells, tmp, how="left", left_on=["Plate", "Well", "Treatment"], right_on=["Metadata_Plate", "Metadata_Well", "Treatment"])
# # # 目的是类似于 sites 一样，以 well 为头进行数据整合
# print(wells)
wells = wells[columns1 + columns2]

wells.to_csv(save_path + "0_noBC_well_level.csv", index=False)


# # 4. BC Sphering
import scipy.linalg
import pandas as pd

class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param=1e-6):
        # Whitening transform on population level data
        self.mu = controls.mean()
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)
        # print(self.mu.shape, self.W.shape)
        
    def whitening_transform(self, X, lambda_, rotate=True):
        C = (1/X.shape[0]) * np.dot(X.T, X)
        s, V = scipy.linalg.eigh(C)
        D = np.diag( 1. / np.sqrt(s + lambda_) )
        W = np.dot(V, D)
        if rotate:
            W = np.dot(W, V.T)
        self.W = W

    def normalize(self, X):
        return np.dot(X - self.mu, self.W)

whN = WhiteningNormalizer(wells.loc[wells["Treatment"] == "DMSO@0", columns2], reg_param=REG_PARAM)

whD = whN.normalize(wells[columns2])

# Save whitened profiles
wells[columns2] = whD

wells.to_csv(save_path + "1_BC_well_level.csv", index=False)


# # 5. Treatment-level profiles / Mean Aggreagation
# Aggregate profiles
columns1 = ["Plate", "Well", "Treatment", "Replicate", "broad_sample"]
columns2 = [i for i in range(num_features)] 

wells = wells.drop(columns=["Plate", "Well"])

# 不要BC的情况
# wells = wells.drop(columns=["Plate", "Well", "broad_sample"])
# print(wells)
profiles = wells.groupby("Treatment").mean().reset_index()
# profiles

# 删除 DMSO 行
# profiles = profiles.drop(profiles.index[-1])
profiles.to_csv(save_path + "2_BC_Treatment_level.csv", index=False)


wells["broad_sample"] = wells["Treatment"].str.split("@", expand=True)[0]

# Recover broad_sample column (cannot be used in groupby because it contains NaN values

tmp = wells.groupby(["Treatment", "broad_sample"])["Replicate"].count().reset_index()
profiles = pd.merge(profiles.reset_index(), tmp, on="Treatment", how="left")
# profiles


profiles = profiles[["Treatment", "broad_sample"] + columns2]
# profiles

# Remove samples without MOA (according to [1])
Y = pd.read_csv(f"data/bbbc022_MOA_MATCHES_official.csv")
# print(Y) # 包含 Treatment 和 Metadata_moa.x 的配对。这才是下游分析任务的最终目标吧
profiles = pd.merge(profiles, Y, left_on="broad_sample", right_on="Var1")
profiles = profiles[["Treatment", "broad_sample", "Metadata_moa.x"] + columns2].sort_values(by="broad_sample")
# profiles

# # 6. Similarity matrix
# Compute Cosine Similarities
COS = sklearn.metrics.pairwise.cosine_similarity(profiles[columns2], profiles[columns2])
# COS.shape

# Transform to tidy format
df = pd.DataFrame(data=COS, index=list(profiles.broad_sample), columns=list(profiles.broad_sample))
#　将行索引重置为默认整数索引，并将原来的行索引 broad_sample 转换为一列，命名为 index。所以，variable　表示原来的broad_sample名。
df = df.reset_index().melt(id_vars=["index"])
# df # 其中每一行都表示 预测的Treatment和 GT 之间的概率。

# Annotate rows
df2 = pd.merge(
    df, 
    profiles[["broad_sample", "Metadata_moa.x"]],  # 为了加 Metadata_moa.x 列数据，先用 broad_sample 建立对应关系，然后删除。
    how="left", 
    left_on="index", # <=== Rows
    right_on="broad_sample"
).drop("broad_sample",axis=1)
# df2

# Annotate columns
#　index　和 variable 是一个东西，都表示 Treatment，但是 index　对应 Metadata_moa.x_x	，variable对应Metadata_moa.x_y
df2 = pd.merge(
    df2, profiles[["broad_sample", "Metadata_moa.x"]],
    how="left", 
    left_on="variable", # <=== Columns
    right_on="broad_sample"
).drop("broad_sample",axis=1)

# df2

# %%
# Rename columns and save
df2.columns = ["Var1", "Var2", "value", "Metadata_moa.x", "Metadata_moa.y"]
df2.to_csv(MATRIX_FILE, index=False)
# df2

# %% [markdown]
# # MOA Evaluation using enrichment analysis

# %%
SIM_MATRIX = MATRIX_FILE  # "data/cos_efn128combinedplatesout_conv6a_1e-2_e30.csv"
OUT_RESUTS = "output/efn128combinedplatesout_conv6a_1e-2_e30"

# %%
def load_similarity_matrix(filename):
    # Load matrix in triplet format and reshape
    cr_mat = pd.read_csv(filename)
    X = cr_mat.pivot(index="Var1", columns="Var2", values="value").reset_index()
    
    # Identify annotations
    Y = cr_mat.groupby("Var1").max().reset_index()
    Y = Y[~Y["Metadata_moa.x"].isna()].sort_values(by="Var1")
    
    # Make sure the matrix is sorted by treatment
    X = X.loc[X.Var1.isin(Y.Var1), ["Var1"] + list(Y.Var1)].sort_values("Var1")
    
    return X,Y

# %%
X, Y = load_similarity_matrix(SIM_MATRIX)  # X 加载了数值, Y 加载了treatment等信息，最后变成随机量。
# Y

# %% [markdown]
# # MOA matching

# %%
Y.groupby("Metadata_moa.x")["Var1"].count()  # 找到每一种 MOA 中 Var1：Treatment 的数量
# Y

# %%
moa_matches = []
Y["Ref_moa"] = Y["Metadata_moa.x"].str.replace('|', '___')  #　potassium channel activator	
# Y['Metadata_moa.x'][63] 

# %%
# MOA 是 Metadata_moa.x 的内部结果，如果 Metadata_moa.x 包含多个预测，则 MOA 中包含多个 True
moa_matches = []
Y["Ref_moa"] = Y["Metadata_moa.x"].str.replace('|', '___')  #　内部包含多项的预测，替换后方便使用正则表达式进行匹配。 'norepinephrine reuptake inhibitor|tricyclic antidepressant'
for k,r in Y.iterrows():
    moas = r["Metadata_moa.x"].split("|")
    # print(moas)
    candidates = []
    for m in moas:
        reg = r'(^|___){}($|___)'.format(m)  
        '''
        正则表达式：
        匹配字符串 m，并确保它要么出现在字符串的开头或结尾，要么被三个下划线分隔。例如，如果 m 是 example，那么生成的正则表达式将是 (^|___)example($|___)，它可以匹配以下情况：
        example 在字符串的开头或结尾。
        example 被 ___ 分隔，如 ___example___。
        '''
        candidates.append(Y["Ref_moa"].str.contains(reg))  # 匹配多项
        # print('reg', reg, candidates[:20])
    matches = candidates[0]
    for c in candidates:
        # print("22", matches, c)
        matches = matches | c
    moa_matches.append(matches)
    # break

moa_matches = np.asarray(moa_matches)
# plt.imshow(moa_matches)


# # Enrichment analysis

# %% [markdown]
# # 输入
# 相似矩阵 (SIM)：一个表示样本或基因之间相似性的矩阵。
# 匹配数据 (moa_matches)：一个包含匹配信息的数据集。
# 阈值 (threshold)：一个数值参数，用于控制分析的严格程度。
# # 输出
# 富集结果：通常是一个包含富集分析结果的列表或数据框，可能包括显著性值、富集分数等。
# 可视化图表：一些函数可能会生成热图、条形图等用于展示富集结果的图表。

results = {}
SIM = np.asarray(X[Y.Var1])
# print("SIM:", SIM.shape)  # (995, 995)
# print(SIM)

is_query = moa_matches.sum(axis=0) > 1 
#　计算 moa_matches 每列的和，并判断是否大于1，结果存储在布尔数组 is_query 中。 大于1：表示该列中至少有两个或更多的非零值。这意味着在 moa_matches 中，该列有多个匹配项。

for i in range(SIM.shape[0]):
    if is_query[i]: #　如果 is_query 中对应位置为 True, 即大于1，有多个匹配项的情况。才能计算富集分析。
        idx = [x for x in range(SIM.shape[1]) if x != i] #　创建一个索引列表 idx，包含除了当前行 i 之外的所有列索引。除开对角线。
        results[i] = enrichment_analysis(SIM[i,idx], moa_matches[i,idx], 99.) # 确认这两个列表中，匹配情况是否高于随即情况
        # 对 SIM 的第 i 行（去掉第 i 列）和 moa_matches 的第 i 行（去掉第 i 列）进行富集分析，并将结果存储在 results 的第 i 个位置。
        if results[i]["ods_ratio"] is np.nan: # ods_ratio大于1 表明SIM[i,idx]中命中的概率高于在 moa_matches[i, idx] 中的概率
            print(results[i]["V"], i)
# results

# 计算并打印富集分析结果中 ods_ratio 的平均值
# 大于 1 则表明： SIM[i, idx] 中，该事件或特征更为显著或富集

folds = [results[x]["ods_ratio"] for x in results] # 提取所有 ods_ratio
enrichment_top_1 = np.mean(folds)
# print("Average folds of enrichment at top 1%:", enrichment_top_1)


enrichment_results = pd.DataFrame(data=results).T
# enrichment_results

# # Average precision analysis

def precision_at_k(sim_matrix, moa_matches, rank_pos=None):
    results = {}
    is_query = moa_matches.sum(axis=0) > 1
    for i in range(sim_matrix.shape[0]):
        if is_query[i]:
            ranking = np.argsort(-sim_matrix[i,idx])
            pk = metrics_precision_at_k(moa_matches[i, ranking[1:]], rank_pos)
            results[i] = {"precision_at_k":pk,"pk":rank_pos}
    return results


positions = [x for x in range(5,55,5)]
average_precision_at_k = []
for pos in positions:
    prec_k = precision_at_k(SIM, moa_matches, pos)
    average_precision_at_k.append(np.mean([prec_k[q]["precision_at_k"] for q in prec_k]))

# plt.figure(figsize=(10,6))
# plt.plot(positions, average_precision_at_k)

top_1percent = max(int(X.shape[0]*0.01), 1)
top_prec = precision_at_k(SIM, moa_matches, top_1percent)
avg_top_prec = np.mean([top_prec[q]["precision_at_k"] for q in top_prec])
# print(f"Average of Precision At Top 1% ({top_1percent} results) => ", avg_top_prec)

prec_at_top1 = pd.DataFrame(data=top_prec).T
# prec_at_top1

# # Recall analysis

def recall_at(sim_matrix, moa_matches, rank_pos=None):
    results = {}
    is_query = moa_matches.sum(axis=0) > 1
    for i in range(sim_matrix.shape[0]):
        if is_query[i]:
            ranking = np.argsort(-sim_matrix[i,:])
            rc = np.sum(moa_matches[i, ranking[1:rank_pos]]) / np.sum(moa_matches[i,:])
            results[i] = {"recall_at_k":rc, "rk":rank_pos}
    return results

recall = []
for pos in positions:
    recall_k = recall_at(SIM, moa_matches, pos)
    recall.append(np.mean([recall_k[x]["recall_at_k"] for x in recall_k]))

# plt.figure(figsize=(10,6))
# plt.plot(positions, recall)

recall_top_10 = recall_at(SIM, moa_matches, top_1percent*10)
avg_recall_at_top = np.mean([recall_top_10[x]["recall_at_k"] for x in recall_top_10])
# print(f"Average Recall At Top 10% ({top_1percent*10} results) => ", avg_recall_at_top)

recall_at_top10 = pd.DataFrame(data=recall_top_10).T

top_1percent = max(int(SIM.shape[0] * 0.01), 1)
top_3percent = max(int(SIM.shape[0] * 0.03), 1)
top_5percent = max(int(SIM.shape[0] * 0.05), 1)
top_10percent = max(int(SIM.shape[0] * 0.10), 1)
top_20percent = max(int(SIM.shape[0] * 0.20), 1)

recall_top_1 = recall_at(SIM, moa_matches, top_1percent)
avg_recall_at_top_1 = np.mean([recall_top_1[x]["recall_at_k"] for x in recall_top_1])
print(f"Average Recall At Top 1% ({top_1percent} results) => ", avg_recall_at_top_1)

recall_top_3 = recall_at(SIM, moa_matches, top_3percent)
avg_recall_at_top_3 = np.mean([recall_top_3[x]["recall_at_k"] for x in recall_top_3])
print(f"Average Recall At Top 3% ({top_3percent} results) => ", avg_recall_at_top_3)

recall_top_5 = recall_at(SIM, moa_matches, top_5percent)
avg_recall_at_top_5 = np.mean([recall_top_5[x]["recall_at_k"] for x in recall_top_5])
print(f"Average Recall At Top 5% ({top_5percent} results) => ", avg_recall_at_top_5)

recall_top_10 = recall_at(SIM, moa_matches, top_10percent)
avg_recall_at_top_10 = np.mean([recall_top_10[x]["recall_at_k"] for x in recall_top_10])
print(f"Average Recall At Top 10% ({top_10percent} results) => ", avg_recall_at_top_10)

# recall_top_20 = recall_at(SIM, moa_matches, top_20percent)
# avg_recall_at_top_20 = np.mean([recall_top_20[x]["recall_at_k"] for x in recall_top_20])
# print(f"Average Recall At Top 20% ({top_20percent} results) => ", avg_recall_at_top_20)



recall_at_top10 = pd.DataFrame(data=recall_top_10).T

# # Interpolated Recall-Precision Curve

recall_axis, average_precision = interpolated_precision_recall_curve(moa_matches, SIM)

# plt.figure(figsize=(10,6))
# plt.plot(recall_axis, average_precision)

# print("Mean Average Precision (MAP): \t", np.mean(average_precision))
# print("Area Under the PR curve: \t", sklearn.metrics.auc(recall_axis, average_precision))

# # Save Results

results = {
    "ranking": positions,
    "precision_at_k": average_precision_at_k,
    "recall": recall,
    "avg_prec@top1": avg_top_prec,
    "avg_recall@top1": avg_recall_at_top,
    "recall_axis": recall_axis,
    "precision_axis": average_precision,
    "mean_average_precision": np.mean(average_precision),
    "reference_library_size": len(X),
    "number_of_queries": len(enrichment_results)
}

with open(OUT_RESUTS + ".pkl", "bw") as out:
    pickle.dump(results, out)

all_results = pd.merge(X["Var1"], enrichment_results, left_index=True, right_index=True)
all_results = pd.merge(all_results, prec_at_top1, left_index=True, right_index=True)
all_results = pd.merge(all_results, recall_at_top10, left_index=True, right_index=True)

all_results.to_csv(OUT_RESUTS + ".csv", index=True)

# result

print("Average folds of enrichment at top 1%:", enrichment_top_1)
print("Mean Average Precision (MAP): \t", np.mean(average_precision))




# -*- coding: utf-8 -*-
"""
随机过程大作业 —— 阶段 0/1-A：数据清洗与物品共现图构建
输出：数据/graph.json（节点、边、布局坐标）、数据/graph_meta.json（统计量）

模型约定（与论文第三章一致）：
  1. 取评分数最多的前 TOP_N 部电影作为状态集 S，筛选规则在论文中明示；
  2. 以用户评分向量的余弦相似度定义边权：
         s_ij = (r_i · r_j) / (||r_i|| ||r_j||)
     只保留 s_ij > 0（余弦非负，保证转移概率非负）；
  3. 每行保留相似度最高的 K 个邻居（Top-K 截断），再对称化为
         w_ij = max(s_ij, s_ji) * 1{i 与 j 互为或单侧 Top-K}
     避免"单向边"导致的非对称权重；
  4. 孤立节点补自环 w_ii = 1，保证行和为正，论文中作为平凡修正说明。
"""
import json
import os

import numpy as np
import pandas as pd
import networkx as nx

ROOT = r"D:\AI\随机过程大作业"
DATA_DIR = os.path.join(ROOT, "数据", "ml-100k")
OUT_DATA = os.path.join(ROOT, "数据")
OUT_WEB = os.path.join(ROOT, "网页", "data")
os.makedirs(OUT_WEB, exist_ok=True)

TOP_N = 200
K = 20

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def load_raw():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep="\t", header=None,
                          names=["u", "i", "r", "ts"])
    cols = ["i", "title", "release", "video", "url"] + [f"g{k}" for k in range(19)]
    items = pd.read_csv(os.path.join(DATA_DIR, "u.item"), sep="|", header=None,
                        names=cols, encoding="latin-1")
    for k, g in enumerate(GENRES):
        items[g] = items[f"g{k}"].astype(int)
    items = items[["i", "title", "release", "url"] + GENRES]
    items["title"] = items["title"].str.strip()
    return ratings, items


def select_items(ratings, items):
    cnt = ratings.groupby("i").size().sort_values(ascending=False)
    keep = cnt.head(TOP_N).index.values
    sub = ratings[ratings["i"].isin(keep)].copy()
    meta = items.set_index("i").loc[keep]
    meta["count"] = cnt.loc[keep].values
    meta["avg_rating"] = sub.groupby("i")["r"].mean().loc[keep].values
    return keep, sub, meta


def build_graph(keep, sub):
    users = sorted(sub["u"].unique())
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {it: k for k, it in enumerate(keep)}
    R = np.zeros((len(users), len(keep)), dtype=np.float64)
    R[sub["u"].map(uidx).values, sub["i"].map(iidx).values] = sub["r"].values

    norms = np.linalg.norm(R, axis=0)
    norms[norms == 0] = 1.0
    S = (R.T @ R) / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)

    # Top-K 截断 + 对称化
    W = np.zeros_like(S)
    for a in range(len(keep)):
        nbr = np.argsort(-S[a])[:K]
        for b in nbr:
            if S[a, b] > 0:
                W[a, b] = max(W[a, b], S[a, b])
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return R, S, W


def main():
    ratings, items = load_raw()
    print(f"[数据] 评分 {len(ratings)} 条，用户 {ratings.u.nunique()}，电影 {ratings.i.nunique()}")
    t0, t1 = ratings.ts.min(), ratings.ts.max()
    print("[数据] 时间范围", pd.to_datetime(t0, unit="s"), "→", pd.to_datetime(t1, unit="s"))

    keep, sub, meta = select_items(ratings, items)
    print(f"[筛选] 取评分数 Top{TOP_N} 电影，覆盖评分 {len(sub)} 条 "
          f"（占 {len(sub)/len(ratings)*100:.1f}%），"
          f"评分数区间 {meta['count'].min()}–{meta['count'].max()}")

    R, S, W = build_graph(keep, sub)

    # 孤立节点补自环
    iso = np.where(W.sum(axis=1) == 0)[0]
    if len(iso):
        W[iso, iso] = 1.0
    print(f"[图] 孤立节点 {len(iso)} 个，已补自环")

    G = nx.from_numpy_array(W)
    n_edges = G.number_of_edges()
    d = dict(G.degree(weight=None))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"[图] 节点 {G.number_of_nodes()}，边 {n_edges}，"
          f"平均度 {np.mean(list(d.values())):.2f}，"
          f"最大度 {max(d.values())}，加权平均度 {np.mean([v for _, v in G.degree(weight='weight')]):.3f}")
    print(f"[图] 连通分量 {len(comps)} 个，最大分量 {len(comps[0])} 个节点，"
          f"其余 {[len(c) for c in comps[1:]]}")
    dens = 2 * n_edges / (TOP_N * (TOP_N - 1))
    print(f"[图] 密度 {dens:.4f}，稀疏度 {1-dens:.4f}，边权范围 "
          f"{W[W>0].min():.4f}–{W.max():.4f}")

    pos = nx.spring_layout(G, weight="weight", seed=42, iterations=300)
    xs = np.array([pos[k][0] for k in range(TOP_N)])
    ys = np.array([pos[k][1] for k in range(TOP_N)])
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)

    nodes = []
    for k, iid in enumerate(keep):
        m = meta.loc[iid]
        nodes.append({
            "id": int(k),
            "mid": int(iid),
            "title": str(m["title"]),
            "count": int(m["count"]),
            "avg": round(float(m["avg_rating"]), 3),
            "year": int(str(m["release"]).split("-")[0]) if str(m["release"])[:4].isdigit() else 0,
            "genres": [g for g in GENRES if int(m[g]) == 1],
            "x": round(float(xs[k]), 4),
            "y": round(float(ys[k]), 4),
        })

    links = []
    for a in range(TOP_N):
        for b in range(a + 1, TOP_N):
            if W[a, b] > 0:
                links.append([a, b, round(float(W[a, b]), 4)])

    graph = {
        "meta": {
            "source": "MovieLens 100K (GroupLens)",
            "n_items_total": int(ratings.i.nunique()),
            "n_users": int(ratings.u.nunique()),
            "n_ratings_total": int(len(ratings)),
            "n_ratings_sub": int(len(sub)),
            "top_n": TOP_N,
            "k_neighbors": K,
            "n_nodes": TOP_N,
            "n_links": len(links),
            "n_components": len(comps),
            "largest_component": len(comps[0]),
            "density": round(dens, 4),
            "iso_nodes": len(iso),
            "time_start": str(pd.to_datetime(t0, unit="s").date()),
            "time_end": str(pd.to_datetime(t1, unit="s").date()),
        },
        "nodes": nodes,
        "links": links,
    }
    with open(os.path.join(OUT_DATA, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT_WEB, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, separators=(",", ":"))

    np.savez_compressed(os.path.join(OUT_DATA, "model.npz"), R=R, S=S, W=W,
                        item_ids=keep, users=np.array(sorted(sub["u"].unique())))
    print("[输出] 数据/graph.json、网页/data/graph.json、数据/model.npz")


if __name__ == "__main__":
    main()

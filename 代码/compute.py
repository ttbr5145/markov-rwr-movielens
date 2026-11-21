# -*- coding: utf-8 -*-
"""
随机过程大作业 —— 阶段 1：带重启随机游走链的数值内核
输入：数据/model.npz（R, W）、数据/ml-100k/u.data、数据/graph.json
输出：数据/results.json、论文/fig/*.pdf、网页/data/timeslices.json

记号：
  P        := 行归一化转移矩阵（Top-K 余弦相似度图，对称化）
  v        := 重启分布（个性化先验，严格正）
  P_alpha  := alpha * P + (1 - alpha) * 1 v^T
  pi       := P_alpha 的唯一平稳分布（个性化 PageRank）

实验协议（避免测试集泄漏）：
  验证集：用时间片 1–2 估 P，预测片 3，选 alpha*；
  测试集：用时间片 1–3 估 P，预测片 4，报告最终指标。
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = r"D:\AI\随机过程大作业"
DATA_DIR = os.path.join(ROOT, "数据", "ml-100k")
OUT_FIG = os.path.join(ROOT, "论文", "fig")
OUT_WEB = os.path.join(ROOT, "网页", "data")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_WEB, exist_ok=True)

ALPHA_THEORY = 0.85
ALPHA_GRID = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 0.95]
EPS = 0.15
TOL = 1e-12
MAX_ITER = 3000
SEED = 0

for cand in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ------------------------------------------------------------------ 工具
def row_normalize(W, fallback):
    s = W.sum(axis=1)
    P = np.zeros_like(W, dtype=float)
    nz = s > 0
    P[nz] = W[nz] / s[nz, None]
    P[~nz] = fallback
    return P


def power_iteration(P, v, alpha, tol=TOL, max_iter=MAX_ITER):
    n = P.shape[0]
    pi = v.copy()
    pi /= pi.sum()
    hist = []
    for k in range(max_iter):
        new = alpha * (pi @ P) + (1 - alpha) * v
        new /= new.sum()
        res = np.abs(new - pi).sum()
        hist.append(res)
        pi = new
        if res < tol:
            return pi, hist, k + 1
    return pi, hist, max_iter


def direct_solve(P, v, alpha):
    n = P.shape[0]
    A = np.eye(n) - alpha * P.T
    b = (1 - alpha) * v
    A[-1, :] = 1.0
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def spectral(P, alpha, v):
    Pa = alpha * P + (1 - alpha) * np.outer(np.ones(P.shape[0]), v)
    ev = np.linalg.eigvals(Pa)
    mag = np.sort(np.abs(ev))[::-1]
    return float(mag[0]), float(mag[1])


def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def gini(p):
    x = np.sort(np.asarray(p))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def kl(p, q, eps=1e-12):
    p = np.maximum(np.asarray(p, float) / np.sum(p), eps)
    q = np.maximum(np.asarray(q, float) / np.sum(q), eps)
    return float((p * np.log(p / q)).sum())


def build_slice_graph(sub, keep, k=20):
    users = sorted(sub["u"].unique())
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(keep)}
    R = np.zeros((len(users), len(keep)))
    R[sub["u"].map(uidx).values, sub["i"].map(iidx).values] = sub["r"].values
    norms = np.linalg.norm(R, axis=0)
    norms[norms == 0] = 1.0
    S = (R.T @ R) / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)
    W = np.zeros_like(S)
    for a in range(len(keep)):
        for b in np.argsort(-S[a])[:k]:
            if S[a, b] > 0:
                W[a, b] = max(W[a, b], S[a, b])
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W, S, R


def top_from(score, rated, k_top=10):
    s = np.array(score, float, copy=True)
    if len(rated):
        s[rated] = -np.inf
    return np.argsort(-s)[:k_top].tolist()


def run_eval(hist_df, test_df, P_rec, S_sim, v_pop, keep, alphas,
             eps=EPS, rel_min=4, min_hist=5, k_top=10):
    n = P_rec.shape[0]
    idx_map = {it: k for k, it in enumerate(keep)}
    hn = hist_df.groupby("u").size()
    tn = test_df.groupby("u").size()
    rel_df = test_df[test_df["r"] >= rel_min]
    rel_cnt = rel_df.groupby("u").size()
    cand = [u for u in sorted(set(hn[hn >= min_hist].index) & set(tn.index))
            if rel_cnt.get(u, 0) >= 1]

    packs = []
    hu = hist_df.groupby("u")
    ru = rel_df.groupby("u")
    for u in cand:
        h = hu.get_group(u)
        rated = np.array([idx_map[i] for i in h["i"].values], dtype=int)
        w = np.maximum(h["r"].values - 2.0, 0.1)
        vu = np.zeros(n)
        np.add.at(vu, rated, w)
        vu /= vu.sum()
        rel = set(idx_map[i] for i in ru.get_group(u)["i"].values)
        packs.append((vu, rated, rel))

    def metrics(lists):
        hit, prec, rec, ndcg = [], [], [], []
        for lst, (_, _, rel) in zip(lists, packs):
            top = set(lst)
            h = len(top & rel)
            hit.append(1.0 if h else 0.0)
            prec.append(h / k_top)
            rec.append(h / len(rel))
            dcg = sum(1.0 / np.log2(i + 2) for i, j in enumerate(lst) if j in rel)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k_top)))
            ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        return {"hit10": float(np.mean(hit)), "p10": float(np.mean(prec)),
                "recall10": float(np.mean(rec)), "ndcg10": float(np.mean(ndcg)),
                "coverage": len({x for l in lists for x in l}) / n}

    out = {"n_users": len(cand),
           "avg_relevant": float(np.mean([len(p[2]) for p in packs])) if packs else 0.0,
           "grid": [], "baselines": {}}
    for a in alphas:
        lists = []
        for vu, rated, _ in packs:
            v = (1 - eps) * vu + eps * v_pop
            p, _, _ = power_iteration(P_rec, v, a)
            lists.append(top_from(p, rated, k_top))
        m = metrics(lists)
        m["alpha"] = a
        out["grid"].append(m)

    for nm in ["pop", "icf", "nr"]:
        lists = []
        for vu, rated, _ in packs:
            if nm == "pop":
                s = v_pop
            elif nm == "icf":
                s = S_sim @ vu
            else:
                v = (1 - eps) * vu + eps * v_pop
                s, _, _ = power_iteration(P_rec, v, 1.0, max_iter=400)
            lists.append(top_from(s, rated, k_top))
        out["baselines"][nm] = metrics(lists)
    return out


# ------------------------------------------------------------------ 主流程
def main():
    np.random.seed(SEED)
    z = np.load(os.path.join(ROOT, "数据", "model.npz"), allow_pickle=True)
    W, keep = z["W"], z["item_ids"]
    n = W.shape[0]

    ratings = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep="\t", header=None,
                          names=["u", "i", "r", "ts"])
    sub = ratings[ratings["i"].isin(keep)].copy()
    counts = sub.groupby("i").size().reindex(keep).fillna(0).values.astype(float)
    v_pop = counts / counts.sum()
    titles = [nd["title"] for nd in json.load(
        open(os.path.join(ROOT, "数据", "graph.json"), encoding="utf-8"))["nodes"]]

    P = row_normalize(W, v_pop)
    res = {"meta": {"alpha_theory": ALPHA_THEORY, "eps": EPS, "tol": TOL,
                    "max_iter": MAX_ITER, "top_n": n,
                    "n_users": int(ratings.u.nunique()),
                    "n_ratings_total": int(len(ratings)),
                    "time_start": str(pd.to_datetime(ratings.ts.min(), unit="s").date()),
                    "time_end": str(pd.to_datetime(ratings.ts.max(), unit="s").date())}}

    # ---------------- alpha 扫描（全局先验 v_pop） ----------------
    alphas = [round(a, 2) for a in np.arange(0.05, 0.96, 0.05)]
    scan = []
    for a in alphas:
        p, h, k = power_iteration(P, v_pop, a)
        _, l2 = spectral(P, a, v_pop)
        scan.append({"alpha": a, "iters": k, "residual": h[-1],
                     "lam2": abs(l2), "gap": 1 - abs(l2),
                     "entropy": entropy(p), "gini": gini(p),
                     "top10_avg_count": float(counts[np.argsort(-p)[:10]].mean())})
    res["alpha_scan"] = scan

    # ---------------- 时间片与逐片图 ----------------
    ts_q = pd.qcut(sub["ts"], 4, labels=False)
    slice_raw, Wt_list, S_sim_list = [], [], []
    for q in range(4):
        s = sub[ts_q == q]
        Wt, St, _ = build_slice_graph(s, keep)
        Gt = nx.from_numpy_array(Wt)
        comps = list(nx.connected_components(Gt))
        slice_raw.append({
            "idx": q + 1,
            "start": str(pd.to_datetime(s.ts.min(), unit="s").date()),
            "end": str(pd.to_datetime(s.ts.max(), unit="s").date()),
            "n": int(len(s)), "n_users": int(s.u.nunique()),
            "n_components": len(comps),
            "largest_component": max(len(c) for c in comps),
            "dangling": int((Wt.sum(axis=1) == 0).sum()),
        })
        Wt_list.append(Wt)
        S_sim_list.append(St)

    # ---------------- 验证集选 alpha ----------------
    hist_v = sub[ts_q <= 1]
    test_v = sub[ts_q == 2]
    Wv, Sv, _ = build_slice_graph(hist_v, keep)
    Pv = row_normalize(Wv, v_pop)
    val = run_eval(hist_v, test_v, Pv, Sv, v_pop, keep, ALPHA_GRID)
    best = max(val["grid"], key=lambda r: (r["ndcg10"], r["hit10"]))
    a_best = best["alpha"]
    print(f"\n[验证集] 用户 {val['n_users']} 个；NDCG@10 最优 alpha = {a_best}")
    for g in val["grid"]:
        print(f"    alpha={g['alpha']:.2f} hit@10={g['hit10']:.3f} "
              f"NDCG@10={g['ndcg10']:.3f} 覆盖率={g['coverage']:.3f}")

    # ---------------- 测试集：用户级 ----------------
    s123 = sub[ts_q < 3]
    s4 = sub[ts_q == 3]
    Wagg, Sagg, _ = build_slice_graph(s123, keep)
    Pagg = row_normalize(Wagg, v_pop)
    tst = run_eval(s123, s4, Pagg, Sagg, v_pop, keep, ALPHA_GRID)
    main_row = next(g for g in tst["grid"] if abs(g["alpha"] - a_best) < 1e-9)
    print(f"\n[测试集] 用户 {tst['n_users']} 个；本模型(alpha={a_best}) "
          f"hit@10={main_row['hit10']:.3f} P@10={main_row['p10']:.3f} "
          f"R@10={main_row['recall10']:.3f} NDCG@10={main_row['ndcg10']:.3f} "
          f"覆盖率={main_row['coverage']:.3f}")
    for k_, v_ in tst["baselines"].items():
        print(f"    基线 {k_:4s} hit@10={v_['hit10']:.3f} NDCG@10={v_['ndcg10']:.3f} "
              f"覆盖率={v_['coverage']:.3f}")

    # ---------------- 主结果：alpha* 下的平稳分布 ----------------
    pi, hist, it = power_iteration(P, v_pop, a_best)
    pi_direct = direct_solve(P, v_pop, a_best)
    l1_gap = float(np.abs(pi - pi_direct).sum())
    Pa = a_best * P + (1 - a_best) * np.outer(np.ones(n), v_pop)
    resid_direct = float(np.abs(pi_direct @ Pa - pi_direct).sum())
    lam1, lam2 = spectral(P, a_best, v_pop)
    top = np.argsort(-pi)[:10]
    res["default"] = {
        "alpha": a_best, "iters": it, "l1_residual": hist[-1],
        "cross_check_l1": l1_gap, "direct_residual": resid_direct,
        "lam1": lam1, "lam2": lam2, "spectral_gap": 1 - abs(lam2),
        "entropy": entropy(pi), "gini": gini(pi),
        "top10": [{"rank": r + 1, "title": titles[j], "pi": round(float(pi[j]), 5),
                   "count": int(counts[j])} for r, j in enumerate(top)],
        "top10_mass": float(pi[top].sum()),
        "vpop_top10_mass": float(v_pop[np.argsort(-v_pop)[:10]].sum()),
        "top10_avg_count": float(counts[top].mean()),
        "vpop_top10_avg_count": float(counts[np.argsort(-v_pop)[:10]].mean()),
    }
    print(f"\n[主结果] alpha={a_best}: {it} 次收敛，L1 残差 {hist[-1]:.2e}；"
          f"双路交叉 L1 差 {l1_gap:.2e}；|lambda2|={abs(lam2):.4f}，谱隙 {1-abs(lam2):.4f}")

    # 个性化扫描（单种子）
    seed_node = int(np.argmax(counts))
    pers = []
    for a in [0.3, 0.5, 0.7, 0.85, 0.95]:
        v_seed = (1 - EPS) * np.eye(n)[seed_node] + EPS * v_pop
        p, _, k = power_iteration(P, v_seed, a)
        tt = np.argsort(-p)[:10]
        pers.append({"alpha": a, "iters": k, "entropy": entropy(p),
                     "top10_avg_count": float(counts[tt].mean()),
                     "top1": titles[tt[0]], "top1_share": float(p[tt[0]])})
    res["personalized_scan"] = {"seed": titles[seed_node], "rows": pers}

    # ---------------- 逐片平稳分布（同一 alpha*） ----------------
    slices, pis = [], []
    for q, d in enumerate(slice_raw):
        Pt = row_normalize(Wt_list[q], v_pop)
        pt, ht, kt = power_iteration(Pt, v_pop, a_best)
        cnt_t = sub[ts_q == q].groupby("i").size().reindex(keep).fillna(0).values.astype(float)
        tt = np.argsort(-pt)[:10]
        pis.append(pt)
        d.update({"iters": kt, "residual": ht[-1], "entropy": entropy(pt),
                  "top10": [titles[j] for j in tt],
                  "top10_avg_count": float(cnt_t[tt].mean()),
                  "coverage_rated": int((cnt_t > 0).sum())})
        slices.append(d)

    # ---------------- 第 4 片：全局分布预测 ----------------
    f4 = s4.groupby("i").size().reindex(keep).fillna(0).values.astype(float)
    f4 = f4 / f4.sum()
    f3 = sub[ts_q == 2].groupby("i").size().reindex(keep).fillna(0).values.astype(float)
    f3 = f3 / f3.sum()
    f123 = s123.groupby("i").size().reindex(keep).fillna(0).values.astype(float)
    b_freq = f123 / f123.sum()

    p_step = a_best * (f3 @ Pagg) + (1 - a_best) * v_pop
    p_step = p_step / p_step.sum()
    p_step85 = ALPHA_THEORY * (f3 @ Pagg) + (1 - ALPHA_THEORY) * v_pop
    p_step85 = p_step85 / p_step85.sum()
    pred_stat = np.mean(pis[:3], axis=0)

    b_norestart, h_nr, it_nr = power_iteration(Pagg, v_pop, 1.0, tol=1e-14, max_iter=5000)
    b_nr2, _, _ = power_iteration(Pagg, np.full(n, 1.0 / n), 1.0, tol=1e-14, max_iter=5000)
    b_icf = Sagg @ b_freq
    b_icf = b_icf / b_icf.sum()

    def evaluate(vec, name):
        t10p = set(np.argsort(-vec)[:10].tolist())
        t10a = set(np.argsort(-f4)[:10].tolist())
        return {"name": name, "kl": kl(f4, vec),
                "hit10": len(t10p & t10a) / 10.0,
                "spearman": float(stats.spearmanr(vec, f4).statistic),
                "top10": [titles[j] for j in np.argsort(-vec)[:10]],
                "top10_avg_count": float(counts[np.argsort(-vec)[:10]].mean())}

    models = [("本模型 一步预测 f3·P_alpha*", p_step),
              ("基线1 纯频次", b_freq),
              ("基线2 无重启平稳分布", b_norestart),
              ("基线3 ItemCF 单步", b_icf),
              ("对照 一步预测 alpha=0.85", p_step85),
              ("对照 平稳分布均值", pred_stat)]
    evals = [evaluate(v_, nm) for nm, v_ in models]

    res["slices"] = slices
    res["prediction"] = {"actual_slice": slices[3],
                         "actual_top10": [titles[j] for j in np.argsort(-f4)[:10]],
                         "models": evals}
    res["no_restart"] = {"iters": it_nr, "converged": bool(h_nr[-1] < 1e-12),
                         "final_residual": h_nr[-1],
                         "l1_between_two_inits": float(np.abs(b_norestart - b_nr2).sum()),
                         "entropy": entropy(b_norestart),
                         "top10_avg_count": float(counts[np.argsort(-b_norestart)[:10]].mean())}
    res["user_level"] = {"alpha_selected": a_best, "validation": val, "test": tst,
                         "main_row": main_row}

    print("\n[第4片预测]")
    for e in evals:
        print(f"    {e['name']:28s} KL={e['kl']:.4f} hit@10={e['hit10']:.2f} "
              f"rho={e['spearman']:.3f} top10均评分数={e['top10_avg_count']:.0f}")

    # ---------------- 网页数据 ----------------
    actual_slices = []
    for q in range(4):
        c = sub[ts_q == q].groupby("i").size().reindex(keep).fillna(0).values.astype(float)
        actual_slices.append([round(float(x), 4) for x in c / c.sum()])
    web = {"meta": json.load(open(os.path.join(ROOT, "数据", "graph.json"),
                                  encoding="utf-8"))["meta"],
           "alpha": a_best,
           "slices": slices,
           "pi": [{"idx": s["idx"], "values": [round(float(x), 4) for x in p]}
                  for s, p in zip(slices, pis)],
           "actual": [{"idx": q + 1, "values": actual_slices[q]} for q in range(4)],
           "global_pi": {"alpha": a_best, "values": [round(float(x), 4) for x in pi]},
           "vpop": [round(float(x), 4) for x in v_pop],
           "gap_by_alpha": {f"{r['alpha']:.2f}": round(r["gap"], 4) for r in scan},
           "prediction": {"models": [{"name": e["name"], "kl": round(e["kl"], 4),
                                      "hit10": round(e["hit10"], 3),
                                      "spearman": round(e["spearman"], 4)} for e in evals],
                          "actual_top10": res["prediction"]["actual_top10"]}}
    with open(os.path.join(OUT_WEB, "timeslices.json"), "w", encoding="utf-8") as f:
        json.dump(web, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(ROOT, "数据", "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in [0.3, 0.5, 0.85, 0.95]:
        _, h, _k = power_iteration(P, v_pop, a)
        ax.semilogy(np.arange(1, len(h) + 1), h, lw=1.6, label=rf"$\alpha={a}$")
    ax.set_xlabel("迭代次数 $k$")
    ax.set_ylabel(r"$L_1$ 残差 $\|\pi_{k+1}-\pi_k\|_1$")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT_FIG, "convergence.pdf"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    xs = [r["alpha"] for r in scan]
    ax1.plot(xs, [r["iters"] for r in scan], "o-", ms=4, color="tab:blue")
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel("迭代次数", color="tab:blue")
    ax1.grid(ls=":", alpha=0.6)
    ax1b = ax1.twinx()
    ax1b.plot(xs, [r["gap"] for r in scan], "s--", ms=3.5, color="tab:red")
    ax1b.set_ylabel(r"谱隙 $1-|\lambda_2|$", color="tab:red")
    ax2.plot(xs, [r["top10_avg_count"] for r in scan], "o-", ms=4)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel("Top-10 平均评分数")
    ax2.grid(ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "alpha_scan.pdf"))
    plt.close(fig)

    o = np.argsort(-pi)[:40]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(Pa[np.ix_(o, o)], cmap="viridis", aspect="auto")
    ax.set_xlabel("状态 j（按 π 降序）")
    ax.set_ylabel("状态 i（按 π 降序）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$P_\alpha(i,j)$")
    fig.savefig(os.path.join(OUT_FIG, "heatmap.pdf"))
    plt.close(fig)

    graph = json.load(open(os.path.join(ROOT, "数据", "graph.json"), encoding="utf-8"))
    gx = np.array([nd["x"] for nd in graph["nodes"]])
    gy = np.array([nd["y"] for nd in graph["nodes"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for a_, b_, w_ in graph["links"]:
        ax.plot([gx[a_], gx[b_]], [gy[a_], gy[b_]], color="0.75", lw=0.35,
                alpha=min(1.0, 0.15 + w_), zorder=1)
    sc = ax.scatter(gx, gy, c=np.log10(pi + 1e-6), s=28 + 260 * pi / pi.max(),
                    cmap="plasma", edgecolors="0.2", linewidths=0.3, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=r"$\log_{10}\pi_i$")
    fig.savefig(os.path.join(OUT_FIG, "network.pdf"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    k15 = 15
    idx_p = np.argsort(-p_step)[:k15]
    yy = np.arange(k15)[::-1]
    ax1.barh(yy + 0.2, f4[idx_p], height=0.38, color="tab:blue", label="真实频次")
    ax1.barh(yy - 0.2, p_step[idx_p], height=0.38, color="tab:orange", label="模型预测")
    ax1.set_yticks(yy)
    ax1.set_yticklabels([titles[j][:22] for j in idx_p], fontsize=6.5)
    ax1.set_xlabel("概率质量")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("按模型预测排序的 Top-15", fontsize=10)
    show4 = [e for e in evals if not e["name"].startswith("对照")]
    yy2 = np.arange(len(show4))[::-1]
    ax2.barh(yy2, [e["hit10"] for e in show4],
             color=["tab:red"] + ["0.6"] * (len(show4) - 1), height=0.5)
    ax2.set_yticks(yy2)
    ax2.set_yticklabels([e["name"].replace(" 一步预测 f3·P_alpha*", "") for e in show4],
                        fontsize=8)
    ax2.set_xlabel("Top-10 命中率")
    ax2.set_xlim(0, 1.0)
    ax2.set_title("第 4 时间片：全局 Top-10 命中率", fontsize=10)
    for i, e in enumerate(show4):
        ax2.text(e["hit10"] + 0.01, len(show4) - 1 - i, f"{e['hit10']:.2f}",
                 va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "timeslice.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.loglog(np.arange(1, n + 1), np.sort(pi)[::-1], "o-", ms=3.5,
              label=rf"π（α={a_best}）")
    ax.loglog(np.arange(1, n + 1), np.sort(v_pop)[::-1], "s--", ms=3, label="评分数占比")
    ax.set_xlabel("状态排名"); ax.set_ylabel("概率质量")
    ax.grid(which="both", ls=":", alpha=0.6)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT_FIG, "longtail.pdf"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    rows = [("本模型", main_row), ("纯热门", tst["baselines"]["pop"]),
            ("ItemCF", tst["baselines"]["icf"]), ("无重启", tst["baselines"]["nr"])]
    yy = np.arange(len(rows))[::-1]
    ax1.barh(yy + 0.2, [r[1]["hit10"] for r in rows], height=0.36, color="tab:red",
             label=r"hit@10")
    ax1.barh(yy - 0.2, [r[1]["ndcg10"] for r in rows], height=0.36, color="tab:blue",
             label=r"NDCG@10")
    ax1.set_yticks(yy)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax1.set_xlim(0, 1.0)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("用户级个性化预测（第 4 时间片）", fontsize=10)
    for i, (_, m) in enumerate(rows):
        ax1.text(m["hit10"] + 0.01, len(rows) - 1 - i + 0.2, f"{m['hit10']:.2f}",
                 va="center", fontsize=7)
        ax1.text(m["ndcg10"] + 0.01, len(rows) - 1 - i - 0.2, f"{m['ndcg10']:.2f}",
                 va="center", fontsize=7)
    g = tst["grid"]
    ax2.plot([r["alpha"] for r in g], [r["hit10"] for r in g], "o-", ms=5, label="hit@10")
    ax2.plot([r["alpha"] for r in g], [r["ndcg10"] for r in g], "s--", ms=4, label="NDCG@10")
    ax2.axvline(a_best, color="0.4", ls=":", lw=1)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel("指标值")
    ax2.set_title(r"$\alpha$ 的探索-利用权衡（测试集）", fontsize=10)
    ax2.grid(ls=":", alpha=0.6)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "user_level.pdf"))
    plt.close(fig)

    print("\n[输出] 数据/results.json, 网页/data/timeslices.json, 论文/fig/*.pdf")


if __name__ == "__main__":
    main()

# 图上随机游走与个性化重启的马尔可夫链模型

以 MovieLens 100K 评分网络为例：把电影共现关系建成加权图，在图上跑一条**带个性化重启的随机游走马尔可夫链**，用同一个对象回答"状态空间怎么划、转移矩阵怎么算、极限分布是否存在且等于平稳分布、如何用历史分布预测未来"这四个问题。

东北林业大学《随机过程》课程大作业。

---

## 一句话结论

重启项 $\alpha$ 让链从"可约、可能不收敛"变成"不可约、非周期、平稳分布唯一"，代价几乎为零；同一个 $P_\alpha$ 的平稳分布既是个性化 PageRank 得分，也是往前推一期的转移核。预测上它赢在**排序质量与覆盖面**（NDCG@10 0.211、覆盖率 0.635），不是"猜最热"（hit@10 与纯热门持平）。

## 方法要点

| 环节 | 做法 |
| --- | --- |
| 状态空间 | 按评分数降序取前 $N=200$ 部电影，覆盖 47,825 条评分（占全量 47.8%），评分数区间 151–583 |
| 边权 | 两物品评分向量的余弦相似度，Top-20 截断后对称化，行归一化得 $P$；悬挂行用先验 $v$ 填充 |
| 重启 | $P_\alpha=\alpha P+(1-\alpha)\mathbf{1}v^\top$，$\alpha^*=0.6$（验证集按 NDCG@10 扫描选出） |
| 先验 | 用户先验正则化 $v_u=(1-\epsilon)\tilde v_u+\epsilon v_{\text{pop}}$，$\epsilon=0.15$ |
| 求 $\pi$ | 幂迭代 40 次，$L_1$ 残差 $9.19\times10^{-13}$；另用直接解线性方程组交叉验证，两者差 $1.13\times10^{-12}$ |
| 预测 | 不做平稳分布外推，而做一步马尔可夫推进 $\hat p_4=f_3P_{\alpha^*}$ |

图结构：200 节点 / 3,028 边 / 平均度 30.28 / 最大度 113 / 密度 0.1522 / 单连通分量。$|\lambda_2|=0.5511$，谱隙 0.4489，$\pi$ 的熵 5.2137。

## 主要结果

**全局分布预测（第 4 时间片）**

| 方法 | KL 散度 | Spearman |
| --- | --- | --- |
| **本模型 PPR ($\alpha^*=0.6$)** | **0.0451** | **0.781** |
| 纯频次（热门） | 0.0585 | 0.642 |
| 无重启随机游走 | 0.1353 | 0.443 |
| ItemCF 单步 | 0.1003 | 0.326 |

KL 比纯频次低 23%，比无重启链与 ItemCF 低一个量级；但 Top-10 命中率 0.30 低于纯热门的 0.60——这是"分布更准"与"头部排序被打散"的必然交换，在评分人数高度集中的数据上是结构性事实。

**用户级个性化预测（104 位合格用户）**

| 方法 | hit@10 | P@10 | R@10 | NDCG@10 | 覆盖率 |
| --- | --- | --- | --- | --- | --- |
| **本模型 PPR ($\alpha^*=0.6$)** | 0.577 | 0.160 | **0.195** | **0.211** | **0.635** |
| 纯热门排序 | **0.577** | 0.150 | 0.184 | 0.200 | 0.365 |
| ItemCF | 0.500 | 0.164 | 0.136 | 0.200 | 0.640 |
| 无重启随机游走 | 0.452 | 0.155 | 0.123 | 0.178 | 0.470 |

NDCG@10 与召回最优，覆盖率比纯热门高 74%；hit@10 打平。

**可约性不是假设出来的**：全量图连通无孤立点，但按时间戳四分位切片后，第 1、2 片裂成 5 个连通分量、多出 4 个悬挂节点，$P^{(t)}$ 在那两片上确实可约。

## 仓库结构

```
论文/
  paper_solo.tex / .pdf     个人版（吴桐，2024213023）
  paper_team.tex / .pdf     团队版（六人，无学号）
  fig/                      7 张插图（network, convergence, alpha_scan, heatmap, longtail, timeslice, user_level）
代码/
  preprocess.py             评分 → 图与 model.npz
  compute.py                model.npz → results.json 与 fig/*.pdf
  build_web.py              内联单文件交互原型
数据/
  graph.json  model.npz  results.json  run_log.txt
网页/
  index.html                离线可运行原型（单文件，无外部依赖）
  template.html
  data/                     graph.json, timeslices.json
```

两版论文正文完全一致，差异仅在封面（班级 / 姓名 / 学号）与页眉右侧（"吴桐 等" vs "吴桐"）。

## 复现

环境：Python 3.13、TeX Live 2025（XeLaTeX）。

```bash
# 1. 取数据（MovieLens 100K，GroupLens）
curl -k -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
# 解压到 数据/ml-100k/

# 2. 构图与计算
python 代码/preprocess.py     # 生成 graph.json 与 model.npz
python 代码/compute.py        # 生成 results.json 与 fig/*.pdf

# 3. 重建网页
python 代码/build_web.py      # 生成 网页/index.html

# 4. 编译论文（各跑两遍）
xelatex 论文/paper_solo.tex   # 个人版
xelatex 论文/paper_team.tex   # 团队版
```

论文所有数字只有一个来源：`数据/results.json`，网页与论文同源；改动计算脚本后重跑第 2 步即可同步。

## 交互原型

单文件 HTML，双击即开，无需联网、无需服务端。左侧 200 节点力导向图，右侧控制台可调 $\alpha$、切换重启开关、选择用户种子，实时看 Top-10 排序、收敛曲线与 $P_\alpha$ 局部热图。

在线版：https://xunsuiwuchu.site/markov/

`网页/index.html` 由 `python 代码/build_web.py` 把 `template.html` 与 `data/*.json` 内联生成，改样式改 `template.html` 后重跑即可。

## 团队

| 项目 | 内容 |
| --- | --- |
| 学院 | 理学院 |
| 专业班级 | 数学与应用数学 2024-02 班、2024-01 班 |
| 授课教师 | 艾晓辉 |
| 成员 | 吴桐（数学二班）、赵睿、张文琦、吴梦豪、方鹏程、于思阳（数学一班） |

## 参考文献

1. Norris J R. *Markov Chains*. Cambridge University Press, 1998.
2. Seneta E. *Non-negative Matrices and Markov Chains*. Springer, 2006.
3. Levin D A, Perron M. *Markov Chains and Mixing Times*. AMS, 2017.
4. Page L, Brin S, Motwani R, Winograd T. The PageRank citation ranking. Stanford InfoLab, 1999.
5. Haveliwala T H. Topic-sensitive PageRank. WWW, 2002.
6. Jeh G, Widom J. SimRank. KDD, 2002.
7. Fouss F, Pirotte A, Renders J M, Saerens M. Random-walk computation of similarities. IEEE TKDE, 2007.
8. Harper F M, Konstan J A. The MovieLens datasets. ACM TiiS, 2015.

---

## 许可

本项目代码、论文与图表采用 [MIT License](LICENSE)，可自由使用、修改与再分发，保留版权声明即可。

MovieLens 100K 数据集不在此列，其版权归 [GroupLens Research](https://grouplens.org/datasets/movielens/) 所有，按原数据集的使用条款使用，仅用于课程学习。

# -*- coding: utf-8 -*-
"""把 graph.json 与 timeslices.json 内联进单文件 index.html，保证离线双击即可运行。"""
import io
import json
import os

ROOT = r"D:\AI\随机过程大作业"
WEB = os.path.join(ROOT, "网页")
tpl_path = os.path.join(WEB, "template.html")
out_path = os.path.join(WEB, "index.html")

graph = json.load(io.open(os.path.join(WEB, "data", "graph.json"), encoding="utf-8"))
sl = json.load(io.open(os.path.join(WEB, "data", "timeslices.json"), encoding="utf-8"))
# 网页只需要必要字段，压缩体积
graph_out = {"meta": graph["meta"],
             "nodes": [{"title": n["title"], "count": n["count"], "x": n["x"], "y": n["y"]}
                       for n in graph["nodes"]],
             "links": graph["links"]}
sl_out = {k: sl[k] for k in ["alpha", "slices", "pi", "actual", "global_pi", "vpop",
                             "prediction", "gap_by_alpha"]}

data = {"graph": graph_out, "slice": sl_out}
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
tpl = io.open(tpl_path, encoding="utf-8").read()
html = tpl.replace("/*__DATA__*/", payload)
io.open(out_path, "w", encoding="utf-8").write(html)
print("index.html 体积：%.1f KB" % (os.path.getsize(out_path) / 1024))
print("节点 %d，边 %d，时间片 %d，alpha=%.2f"
      % (len(graph_out["nodes"]), len(graph_out["links"]), len(sl_out["slices"]), sl_out["alpha"]))

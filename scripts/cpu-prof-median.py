# Per-graph medians from a GGML_CPU_PROF=1 log (the llama.cpp fork's CPU profiler prints cumulative per-graph averages
# after every graph). Means over a window of graphs are skewed by the few large graphs that share the node-count range,
# e.g. the engine's startup calibration prefills on the decode context; medians of the per-graph increments are not.
#   python cpu-prof-median.py <stderr log> [min_nodes max_nodes] [n_ops]
import os, re, sys, statistics as st
from collections import defaultdict
path=sys.argv[1]; lo=int(sys.argv[2]) if len(sys.argv)>2 else 800; hi=int(sys.argv[3]) if len(sys.argv)>3 else 2000
dumps=[]; cur=None
for l in open(path, encoding='utf-8', errors='replace'):
    m=re.search(r'\[cpu-prof\] graphs (\d+) \| per graph: nodes (\d+), work ([\d.]+) ms, barrier-wait ([\d.]+) ms', l)
    if m:
        cur={'G':int(m.group(1)),'nodes':int(m.group(2)),'work':float(m.group(3)),'wait':float(m.group(4)),'ops':{},'mm':{}}; dumps.append(cur); continue
    if cur is None: continue
    m=re.search(r'\[cpu-prof\]   (\S+)\s+n/graph\s+([\d.]+)\s+work\s+([\d.]+) ms\s+wait\s+([\d.]+) ms', l)
    if m: cur['ops'][m.group(1)]=(float(m.group(2)),float(m.group(3)),float(m.group(4))); continue
    m=re.search(r'\[cpu-prof\]   mm (\S+)\s+(\S+)\s+(\d+) x (\d+)\s+n/graph\s+([\d.]+)\s+([\d.]+) ms/graph \(wait ([\d.]+)\)\s+([\d.]+) MB', l)
    if m: cur['mm'][f"{m.group(1)} {m.group(2)} {m.group(3)}x{m.group(4)}"]=(float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8)))
acc=defaultdict(list); prev=None
for d in dumps:
    if prev is None or d['G']!=prev['G']+1: prev=d; continue
    G=d['G']; P=prev['G']; inc=lambda a,b: a*G-b*P
    nodes=inc(d['nodes'],prev['nodes'])
    if lo<=nodes<=hi:
        acc['nodes'].append(nodes); acc['work'].append(inc(d['work'],prev['work'])); acc['wait'].append(inc(d['wait'],prev['wait']))
        for k,v in d['ops'].items():
            pv=prev['ops'].get(k,(0,0,0)); acc['op '+k].append(inc(v[1],pv[1])+inc(v[2],pv[2]))
        for k,v in d['mm'].items():
            pv=prev['mm'].get(k,(0,0,0,0)); acc['mm '+k].append(inc(v[1],pv[1])); acc['mmB '+k].append(inc(v[3],pv[3]))
    prev=d
n=len(acc['work']); md=lambda k: st.median(acc[k]) if acc[k] else 0
print(f"{os.path.basename(path)}: {n} graphs, median: nodes {md('nodes'):.0f} work {md('work'):.3f} ms wait {md('wait'):.3f} ms  (mean work {st.mean(acc['work']) if n else 0:.3f})")
for k in sorted([k for k in acc if k.startswith('op ')], key=lambda k:-md(k))[:int(sys.argv[4]) if len(sys.argv)>4 else 6]: print(f"  {k[3:]:18s} {md(k):7.3f} ms")
for k in sorted([k for k in acc if k.startswith('mm ')], key=lambda k:-md(k)):
    t=md(k); B=md('mmB '+k[3:]); print(f"  mm {k[3:]:26s} {t:7.3f} ms {B:7.2f} MB {B/t if t else 0:6.1f} GB/s")

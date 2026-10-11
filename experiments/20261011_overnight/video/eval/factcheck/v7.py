import sys, numpy as np
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L
from course import heading, r1_anchors
tot=0
for log in ['20261010-203924','20261010-204208','20261010-204605']:
    l=L(log); heading(l); an=r1_anchors(l)
    ai=[a for a in an if l.sel_at(a) and l.sel_at(a+1.0)]
    print(l.name,'R1 starts',[round(a,2) for a in an],'AI-driven',[round(a,2) for a in ai]); tot+=len(ai)
print('AI R1 passages',tot)

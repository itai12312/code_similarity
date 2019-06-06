import pandas as pd
from collections import defaultdict
df = pd.read_csv('../codes_short/results1.csv')
d = defaultdict(lambda:0, {})
for name in df['qName'].values:
    d[name] += 1
a = sorted(d.items(), key=lambda x:-x[1])
counts = 0
all = 0
for item in a:
    if 'XSS' in item[0] or 'SQL' in item[0]:
        counts += item[1]
    all += item[1]
    print(item)
print(counts)
print(all)

in_filename = '../codes/results.csv'
out_filename = '../codes/results1.csv'

from functools import partial
chunksize = 100*1024*1024 # read 100MB at a time

# Decode with UTF-8 and replace errors with "?"
with open(in_filename, 'rb') as in_file:
    with open(out_filename, 'w') as out_file:
        for byte_fragment in iter(partial(in_file.read, chunksize), b''):
            out_file.write(byte_fragment.decode(encoding='utf_8', errors='replace'))

# Now read the repaired file into a dataframe
import pandas as pd
df = pd.read_csv(out_filename)

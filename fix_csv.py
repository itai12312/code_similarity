import io

with io.open('../codes/results.csv','r',encoding='utf-8',errors='ignore') as infile, \
    io.open('../codes/results1.csv','w',encoding='utf-8',errors='ignore') as outfile:
    for line in infile:
        print(*line.split(), file=outfile)

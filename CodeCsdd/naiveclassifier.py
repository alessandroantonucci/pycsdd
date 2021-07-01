# Implements a naive classifier that predicts c[i] = argmax P(c[i],n[i])

import sys

if len(sys.argv) < 3:
	print ("Usage: python", sys.argv[0], "train test")
	exit(1)
RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

# read training  data
print(RED)
print("Learning counts from data...")
with open(sys.argv[1]) as f:
    header = f.readline().strip('\n').split(',')
    classes = header[0:len(header)//2]
    features = header[len(header)//2:]
    counts = { (c,f): { (0,0):0, (0,1):0, (1,0):0, (1,1): 0 } for (c,f) in zip(classes,features) }
    for line in f:
        line = [ int(x) for x in line.split(',') ]
        for i in range(len(classes)):
            counts[( classes[i], features[i] )][(line[i],line[i+len(classes)])] += 1
        
print("Classifying test set...")

acc = [0,0,0,0,0,0,0]
total = [0,0,0,0,0,0,0]
with open(sys.argv[2]) as f:
    header = f.readline()
    for line in f:
        line = [ int(x) for x in line.split(',') ]
        for i in range(len(classes)):
            c0 = counts[( classes[i], features[i] )][(0,line[i+len(classes)])]
            c1 = counts[( classes[i], features[i] )][(1,line[i+len(classes)])]
            if c0 > c1:
                # predict c[i] = 0
                if line[i] == 0:
                    acc[i] += 1
            else:
                # predict c[i] = 1
                if line[i] == 1:
                    acc[i] += 1
            total[i] += 1

print(NOCOLOR)    

for i in range(len(classes)):
    print(classes[i], end = " ")
    print("Accuracy =   {:3.2f}%  ({}/{})".format(100*acc[i]/total[i],acc[i],total[i]))

print(GREEN)
print("-----------------------------------------------")
print("Overall Accuracy =   {:3.2f}%  ({}/{})".format(100*sum(acc)/sum(total),sum(acc),sum(total)))
print(NOCOLOR)
#print(counts)

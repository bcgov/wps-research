'''deprecated / not in use: script was for plotting downloads per day'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv
lines = [x.strip().split() for x in os.popen("ls -latrh sentinel2_google").readlines()]
num = {'Jan': 0,
       'Feb': 1,
       'Mar': 2,
       'Apr': 3,
       'May': 4,
       'Jun': 5,
       'Jul': 6,
       'Aug': 7,
       'Sep': 8,
       'Oct': 9,
       'Nov':10,
       'Dec':11}

ci = 0
count = {}
for line in lines:
    fn = line[-1]
    day, month = None, None
    try:
        day = line[-3].zfill(2)
        month = str(str(num[line[-4]] + 1).zfill(2))
    except:
        pass
    if fn[-5:] == '.SAFE':
        s = month + day

        if s not in count:
            count[s] = 0
        count[s] += 1

        # print(month, day, line)
        ci += 1
        if ci > 2:
            pass # sys.exit(1)

x, y = [], []
for c in count:
    x.append(int(c))
    y.append(int(count[c]))
    
plt.figure()
plt.plot(x,y, marker='x', label='frames/day')
mean = np.mean(np.array(y))
print('mean', mean)
plt.plot(x, [mean for i in y], label='mean')
plt.xlim([615, 825])
plt.legend()
plt.show()

print(count)
print(list(count))

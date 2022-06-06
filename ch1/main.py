import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")

x = data[:, 0]
y = data[:, 1]

#print(sum(np.isnan(y)))

y = y[~np.isnan(x)]
x = x[~np.isnan(x)]

plt.scatter(x, y, s=5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], 
        [f'week {w}' for w in range(10)])

plt.autoscale(tight=True)
plt.grid()
plt.show()


fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

import pylab as pl
import numpy as np

# d = {"GNB": 77.85, "BNB": 59.76, "SVM": 79.88, "RF": 79.09, "DT": 79.01}
d = {"Sex": 9.87, "Age": 0.72, "Pclass": -0.2, "Fare": -1.08, "SibSp": 1.53, "Parch": 0.93}
X = np.arange(len(d))
pl.bar(X, d.values(), align='center', width=0.5)
pl.xticks(X, d.keys())
pl.plot([-0.5, 5.5], [0, 0], "k--")
ymax = max(d.values()) + 1
pl.ylim(-2, ymax)
pl.xlabel('Attribute')
pl.ylabel('Influence')
pl.show()
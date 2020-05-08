import sqlite3
import numpy as np
#import pandas as np
from scipy import stats
#import statsmodels.api as sm
#from statsmodels.tools import eval_measures
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

objects = ('Simple', 'Predicate', 'Adjunct', 'Binding', 'Question', 'Auxiliary')
y_pos = np.arange(len(objects))
performance = [13, 43, 39, 6, 6, 19]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Occurrences')
plt.xlabel('Sentence Type')

plt.show()
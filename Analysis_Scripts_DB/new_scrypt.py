
import sqlite3
import numpy as np
#import pandas as np
from scipy import stats
#import statsmodels.api as sm
#from statsmodels.tools import eval_measures
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
TOTAL=200

conn = sqlite3.connect('annotations.db')
c = conn.cursor()
query = "SELECT annotations.Sentence, annotations.Acceptability, annotations.Simple, annotations.Predicate, annotations.Adjunct, annotations.Imperative, annotations.Binding, annotations.Question, annotations.Auxiliary FROM annotations;"

c.execute(query)
conn.commit()
total_simple = 0
total_pred = 0
total_auxiliary = 0
total_adjunct = 0
total_imperative = 0
total_binding = 0
total_question = 0

# Adjunct, Argument, Imperative, Binding, Question, Auxiliary

total_correct = 0
for row in c :
	sentence = row[0]
	acceptability = row[1]
	simple = int(row[2])
	predicate = int(row[3])
	adjunct = int(row[4])
	imperative = int(row[5])
	binding = int(row[6])
	question = int(row[7])
	auxiliary = int(row[8])
	#print(predicate, simple)
	if simple == 1:
		total_simple += 1
	elif predicate == 1:
		total_pred += 1
	elif adjunct == 1:
		total_adjunct += 1
	elif imperative == 1:
		total_imperative += 1
	elif binding == 1:
		total_binding += 1
	elif question == 1:
		total_question += 1
	elif auxiliary == 1:
		total_auxiliary += 1



objects = ('Simple', 'Predicate', 'Adjunct', 'Binding', 'Question', 'Auxiliary')
y_pos = np.arange(len(objects))
counts = [total_simple, total_pred, total_adjunct, total_binding, total_question, total_auxiliary]

plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Occurrences')
plt.xlabel('Sentence Type')

plt.show()


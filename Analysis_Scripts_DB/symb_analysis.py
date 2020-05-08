
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



query = "SELECT annotations.Sentence, annotations.Acceptability, annotations.Simple, annotations.Predicate, annotations.Adjunct, annotations.Imperative, annotations.Binding, annotations.Question, annotations.Auxiliary, symb_results.prediction FROM annotations INNER JOIN symb_results ON symb_results.sentence = annotations.Sentence;"

c.execute(query)
conn.commit()
num_correct_simple = 0
total_simple = 0

num_correct_predicate = 0
total_pred = 0

num_correct_auxiliary = 0
total_auxiliary = 0

num_correct_adjunct= 0
total_adjunct = 0

num_correct_imperative= 0
total_imperative = 0

num_correct_binding = 0
total_binding = 0

num_correct_question = 0
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
	prediction = row[9]
	#print(predicate, simple)
	if prediction == acceptability :
		total_correct += 1

	if simple == 1:
		if prediction == acceptability :
			num_correct_simple += 1
		total_simple += 1
	elif predicate == 1:
		if prediction == acceptability :
			num_correct_predicate += 1
		total_pred += 1
	elif adjunct == 1:
		if prediction == acceptability :
			num_correct_adjunct +=1
		total_adjunct += 1
	elif imperative == 1:
		if prediction == acceptability :
			num_correct_imperative += 1
		total_imperative += 1
	elif binding == 1:
		if prediction == acceptability :
			num_correct_binding += 1
		total_binding += 1
	elif question == 1:
		if prediction == acceptability :
			num_correct_question += 1
		total_question += 1
	elif auxiliary == 1:
		if prediction == acceptability :
			num_correct_auxiliary+= 1
		total_auxiliary += 1

percent_correct_simple = float(num_correct_simple) / float(total_simple)
percent_incorrect_simple = 1 - percent_correct_simple
print("percent correct simple", round(percent_correct_simple, 3), total_simple)

percent_correct_predicate = float(num_correct_predicate) / float(total_pred)
percent_incorrect_predicate= 1 - percent_correct_predicate
print("percent correct predicate", round(percent_correct_predicate, 3), total_pred)

percent_correct_adjunct = float(num_correct_adjunct) / float(total_adjunct)
percent_incorrect_adjunct = 1 - percent_correct_adjunct
print("percent correct adjunct", round(percent_correct_adjunct, 3), total_adjunct)

total_imperative = 1
percent_correct_imperative = float(num_correct_imperative) / float(total_imperative)
percent_incorrect_imperative = 1 - percent_correct_imperative
#print("percent correct imperative", percent_correct_imperative)

percent_correct_binding = float(num_correct_binding) / float(total_binding)
percent_incorrect_binding = 1 - percent_correct_binding
print("percent correct binding", round(percent_correct_binding, 3), total_binding)

percent_correct_question = float(num_correct_question) / float(total_question)
percent_incorrect_question = 1 - percent_correct_question
print("percent correct question", round(percent_correct_question, 3), total_question)

percent_correct_auxiliary = float(num_correct_auxiliary) / float(total_auxiliary)
percent_incorrect_auxiliary = 1 - percent_correct_auxiliary
print("percent correct auxiliary", round(percent_correct_auxiliary, 3), total_auxiliary)

labels = 'correctly\n classified', 'incorrectly\n classified'
sizes_simple = [percent_correct_simple, percent_incorrect_simple]
sizes_predicate = [percent_correct_predicate, percent_incorrect_predicate]
sizes_adjunct = [percent_correct_adjunct, percent_incorrect_adjunct]
sizes_imperative = [percent_correct_imperative, percent_incorrect_imperative]
sizes_binding = [percent_correct_binding, percent_incorrect_binding]
sizes_question = [percent_correct_question, percent_incorrect_question]
sizes_auxiliary = [percent_correct_auxiliary, percent_incorrect_auxiliary]

fig1, ax1 = plt.subplots()
ax1.pie(sizes_simple, labels=labels)
ax1.set_title("Simple Sentences")

fig2, ax2 = plt.subplots()
ax2.pie(sizes_predicate, labels=labels)
ax2.set_title("Predicate Sentences")

fig3, ax3 = plt.subplots()
ax3.pie(sizes_adjunct, labels=labels)
ax3.set_title("Adjunct Sentences")

fig4, ax4 = plt.subplots()
ax4.pie(sizes_imperative, labels=labels)
ax4.set_title("Imperative Sentences")

fig5, ax5 = plt.subplots()
ax5.pie(sizes_binding, labels=labels)
ax5.set_title("Binding Sentences")

fig6, ax6 = plt.subplots()
ax6.pie(sizes_question, labels=labels)
ax6.set_title("Questions")

fig7, ax7 = plt.subplots()
ax7.pie(sizes_auxiliary, labels=labels)
ax7.set_title("Auxiliary Sentences")

#fig8, ax8 = plt.subplots()
#x = ("Simple", "Predicate", "Adjunct", "Binding", "Questions", "Auxiliary")
#counts = [total_simple, total_pred, total_adjunct, total_binding, total_question, total_auxiliary]

#ax8.bar(x, counts, color='green')


plt.show()






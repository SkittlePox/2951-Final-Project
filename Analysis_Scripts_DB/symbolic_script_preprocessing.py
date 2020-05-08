f = open("symb-sum-norm-test.txt","r")

out = open("out_symb.tsv", "w+")
count = 0

#neur_format = ["sentence", "perplexity", "CoLA Label", "Sentence Length"]
out.write("sentence\tprob\tCoLALabel\tprediction\tSentence_Length\n")

for line in f:
	line = line.split("\t")
	line[0] = line[0][:-2] + line[0][-1]
	pred = 0
	if (float(line[2]) >= -0.635) :
		pred = 1
	out.write(line[0] + "\t" + line[2] + "\t" + line[1] + "\t" + str(pred) + "\t" + line[3])
	count += 1
	#print(len(line))
    #f.write("This is line %d\r\n" % (i+1))

print(count)

#SELECT annotations.Sentence, annotations.Acceptability, annotations.Simple, annotations_neur.CoLALabel FRations INNER JOIN annotations_neur ON annotations_neur.sentence = annotations.Sentence;

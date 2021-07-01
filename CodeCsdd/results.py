import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../Data/results_exp3.csv')

if False:
	k = 1
	for ts in [10, 15, 20, 50, 100]: #ts in enumerate([10,15,20,50]):
		data2 = data[data['train_size']==ts]
		x = data2['p_f'].tolist()
		y1 = data2['psdd_acc'].tolist()
		y2 = data2['hmm_acc'].tolist()
		y3 = data2['csdd_u80'].tolist()
		y4 = data2['ihmm_u80'].tolist()
		plt.subplot(1,5,k)
		plt.plot(x, y1, 'r-', x, y2, 'b-', x, y3, 'r--', x, y4, 'b--')
		plt.ylabel('Accuracy')
		plt.xlabel('Probability of failure')
		plt.title(ts)
		plt.ylim((0.75, 1.0))
		k += 1
		for a,b,c,d,e in zip(x,y1,y2,y3,y4):
			print('%2.4f,%2.4f,%2.4f,%2.4f,%2.4f' %(a,b,c,d,e))
	plt.savefig('accuracies.png')
	plt.show()

if True:
	k = 1
	for ts in [10, 15, 20, 50, 100]: #ts in enumerate([10,15,20,50]):
		data2 = data[data['train_size']==ts]
		x = data2['p_f'].tolist()
		y1 = data2['psdd_acc'].tolist()
		y2 = data2['psdd_det_acc'].tolist()
		y3 = data2['psdd_indet_acc'].tolist()
		y4 = data2['csdd_det'].tolist()
		for a,b,c,d,e in zip(x,y1,y2,y3,y4):
			print('%2.4f,%2.4f,%2.4f,%2.4f,%2.4f' %(a,b,c,d,e))
		plt.subplot(1,5,k)
		plt.plot(x, y1, 'r--', x, y2, 'g-', x, y3, 'b-', x,y4, 'r*')
		plt.ylabel('Accuracy')
		plt.xlabel('Probability of failure')
		plt.title(ts)
		plt.ylim((0.3, 1.1))
		k += 1
	plt.show()

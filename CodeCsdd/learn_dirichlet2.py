# learn non-stationary precise hmm model using Dirichlet prior
# assumes constant number of states and symbols in each step
import sys
from array import array

#if len(sys.argv) < 7:
#	print "Usage: python", sys.argv[0], "N M K observations_filename tags_filename model_filename"
#	exit(1)
def dirichlet(N, M, K, observations_filename, tags_filename, model_filename, ess):
#N=int(sys.argv[1]) # nr. of states
#M=int(sys.argv[2]) # nr. of possible observations/symbols
#K=int(sys.argv[3]) # number of steps
# uncomment to choose different prior
	prior = "laplace"
	sN = ess*1.0*N # laplace priors
	sM = ess*1.0*M
	#prior = "perks"
	#sN = 0.5 # Perks priors
	#sM = 0.5
	#sN = 1.0 # Perks priors
	#sM = 1.0
	#sN = 2.0 # Perks priors
	#sM = 2.0
	vN = 1.0/N
	vM = 1.0/M
	#prior = "jeffreys"
	#sN = N/2.0 # Jeffreys priors
	#sM = M/2.0
	
	#print('N:', N)
	#print('M:', M)
	#print('K:', K)
	
	a = [[] for k in range(K-1)] # transition probabilities
	b = [[] for k in range(K)] # emission probabilities
	p = [] # prior probabilities
	#O = [[0 for j in range(M)] for k in range(K)] # to count unobserved symbols
	
	step = int(0.2*N*K)+1 # show amount of initialized array at % steps
	
	#print('initializing count arrays... ')
	# assign zeros to all possible combinations
	for k in range(K):
		for i in range(N):
			#if i % step == 0:
			#	print('%d%% ..' % (int(i*k*100.0/N/K)),sys.stdout.flush())
			if k > 0:
				a[k-1].append( array('I',[0 for j in range(N)]) )
			b[k].append( array('I',[0 for j in range(M)]) )
	#print('100%')
	p = [0 for j in range(N)]
	
	# read training  data
	o = open(observations_filename,'r').read().splitlines() # observations file
	q = open(tags_filename,'r').read().splitlines() # states file
	
	# compute counts from data
	#if len(o) != len(q):
	#	print('Dataset dimension mismatch!')
	#	exit(-1)
	
	#print('computing counts...')
	step = int(0.15*len(o))+1 # show amount of sequences read at % steps
	for i in range(len(o)): # for each sequence in dataset
		#if i % step == 0:
		#	print('%d%% ..' % (int(i*100.0/len(o))),sys.stdout.flush())
		if o[i][0] != "#":
			tokens = o[i].split()
			tags = q[i].split()
			#if len(tokens) != K or len(tags) != K:
			#	print('Sequence length mismatch!', 'Sequence:', i)
			#	exit(-1)
			# increase prior probability count
			# ignore states grater or equal than N
			p[int(tags[0])]+=1
			# ignore states and observations not in range
			b[0][int(tags[0])][int(tokens[0])] += 1
			#O[0][int(tokens[0])] += 1
	
			# increase emission and transition prob count for each token in seq
			for t in range(1,K):
				a[t-1][int(tags[t-1])][int(tags[t])] += 1
				# ignore states and observations not in range
				b[t][int(tags[t])][int(tokens[t])] += 1
					#O[t][int(tokens[t])] += 1
	
	#print('100%')
	
	# compute nr. of unseen symbols
	## unseen = [0 for k in range(K)]
	## for k in range(K):
	## 	for j in range(M):
	## 		if O[k][j] == 0:
	## 			unseen[k] += 1
	## 	print unseen[k], "unseen words in step", k
	
	#print('saving model to file...')
	# save model to file
	out = open(model_filename,'w')
	# header
	out.write(str(K) + '\n')
	for k in range(K):
		out.write(str(N) + ' ')
	out.write('\n')
	for k in range(K):
		out.write(str(M) + ' ')
	out.write('\n')
	# lower transition probabilities
	for k in range(K-1):
		for i in range(N):
			Ni = 0.0
			for j in range(N):
				Ni += a[k][i][j]
			
			for j in range(N):
				prob = 1.0*(a[k][i][j]+vN*sN)/(Ni+sN)
				out.write(str(prob) + ' ')
	out.write('\n')
	# upper transition probabilities
	for k in range(K-1):
		for i in range(N):
			Ni = 0.0
			for j in range(N):
				Ni += a[k][i][j]
			
			for j in range(N):
				prob = 1.0*(a[k][i][j]+vN*sN)/(Ni+sN)
				out.write(str(prob) + ' ')
	out.write('\n')
	# lower emission probabilities
	for k in range(K):
		for i in range(N):
			Ni = 0.0
			for j in range(M):
				Ni += b[k][i][j]
			
			for j in range(M):
				prob = 1.0*(b[k][i][j]+vM*sM)/(Ni+sM)
				out.write(str(prob) + ' ')
	out.write('\n')
	# upper emission probabilities
	for k in range(K):
		for i in range(N):
			Ni = 0.0
			for j in range(M):
				Ni += b[k][i][j]
			
			for j in range(M):
				prob = 1.0*(b[k][i][j]+vM*sM)/(Ni+sM)
				out.write(str(prob) + ' ')
	out.write('\n')
	# lower prior probabilities
	Ni = 0
	for i in range(N):
		Ni += p[i]
	for i in range(N):
		prob = 1.0*(p[i]+vN*sN)/(Ni+sN)
		out.write(str(prob) + ' ')
	out.write('\n')
	# upper prior probabilities
	for i in range(N):
		prob = 1.0*(p[i]+vN*sN)/(Ni+sN)
		out.write(str(prob) + ' ')
	out.write('\n')
	out.close()

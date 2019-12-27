import copy
import pandas as pd
from pulp import *
from tqdm import tqdm
from node import DecisionNode, TerminalNode
from vtree_model import Vtree2

def sgn(x):
	y = 0
	if x > 0:
		y = 1
	return y


class Csdd:

	def __init__(self):
		self.opt = -2  # -1 = min CSDD / 0 = PSDD / 1 max = CSDD
		self.problem = 0  # 0 = marginal, 1 = conditional, 2 = MAP, 3 = Robust
		self.nodes = {}  # Array of nodes
		self.mu = -1.0  # mu (variable used for conditionals)
		self.vtree = None  # vtree
		self.children = {} # vtree information
		self.sdd = None  # sdd ? REMOVE?
		self.root = -1  # root of the sdd
		self.psdd = None
		self.xstar = []  # Same size of the total number of variables
		self.evidence = []  # Array of evidences
		self.literals = []  # Labels with the literals
		self.logical_evidence = []
		self.csv = []
		self.serial = ''

	def set_csv(self, csv): # Read the csv and set the column headers as literals
		self.csv = pd.read_csv(csv, delimiter=',', header=0)
		self.literals = list(self.csv.columns)
	
	def set_vtree(self, filename): # Read the vtree
		p = dict()
		self.vtree, my_list = Vtree2.read(filename)
		n_inner_nodes = len([l for l in my_list if str(l)[0] == 'I']) # n of inner nodes
		# Start from the root and split left and right variables
		active = self.vtree
		self.children[active.id] = [list(active.left.variables()), list(active.right.variables())]
		# Iterate unless all the decision nodes are processed
		while len(list(self.children.keys())) < n_inner_nodes:  # DFS
			if str(active.left)[0] == 'I' and active.left.id not in list(self.children.keys()):
				p[active.left.id] = active
				active = active.left
				self.children[active.id] = [list(active.left.variables()), list(active.right.variables())]
			# if left already processed, go down-right
			elif str(active.right)[0] == 'I' and active.right.id not in list(self.children.keys()):
				p[active.right.id] = active
				active = active.right
				self.children[active.id] = [list(active.left.variables()), list(active.right.variables())]
			else:
				active = p[active.id]  # Go up (both children processed)

	def set_left_right(self):
			for node in self.nodes:
				if self.nodes[node].kind == 1:
					self.nodes[node].left = self.children[self.nodes[node].vtree][0]
					self.nodes[node].right = self.children[self.nodes[node].vtree][1]

	def set_sdd(self, filename):	# Read Sdd structure from a file
		
		with open(filename, 'r') as f:
			sdd_lines = f.readlines()
		
		for literal in [_[1:].strip() for _ in sdd_lines if _[0] == 'L']:
			# DEBUG: Decide a single format
			if len(literal.split(' ')) == 3:
				node_id, _, lit = [int(_) for _ in literal.split(' ')]
			else:
				node_id, lit = [int(_) for _ in literal.split(' ')]
			self.nodes[node_id] = TerminalNode(node_id, abs(lit), sgn(lit))
		
		for decision in [_[1:].strip() for _ in sdd_lines if _[0] == 'D']:
			dd = [int(_) for _ in decision.split(' ')]
			self.nodes[dd[0]] = DecisionNode(dd[0], dd[1], dd[3::2], dd[4::2])
		
		for bot_node in [int(_[1:].strip()) for _ in sdd_lines if _[0] == 'F']:
			for _, node in self.nodes.items():
				if node.kind == 1:
					if bot_node in node.primes:
						vtree_par = self.nodes[node.id].vtree
						vtree_child = self.children[vtree_par][0][0]
						break
					if bot_node in node.subs:
						vtree_par = self.nodes[node.id].vtree
						vtree_child = self.children[vtree_par][1][0]
						break
			self.nodes[bot_node] = TerminalNode(bot_node, vtree_child, 2, (vtree_child - 1) * 2)
		
		for top_node in [int(_[1:].strip()) for _ in sdd_lines if _[0] == 'T']:
			for _, node in self.nodes.items():
				if node.kind == 1:
					if top_node in node.primes:
						vtree_par = self.nodes[node.id].vtree
						vtree_child = self.children[vtree_par][0][0]
						break
					if top_node in node.subs:
						vtree_par = self.nodes[node.id].vtree
						vtree_child = self.children[vtree_par][1][0]
						break
			self.nodes[top_node] = TerminalNode(top_node, vtree_child, 3, (vtree_child - 1) * 2)
	
	def set_root(self):			# Find the root of a sdd
		
		descendants = [self.nodes[_].primes + self.nodes[_].subs for _ in self.nodes if self.nodes[_].kind == 1]
		descendants = [item for sublist in descendants for item in sublist]
		decision_nodes = [_ for _ in self.nodes if self.nodes[_].kind == 1]
		self.root = [_ for _ in decision_nodes if _ not in descendants][0]
	
	def learn_counts(self):	# Exctract counts from the csv
		
		# Decision nodes
		for decision_node in tqdm([_ for _ in self.nodes if self.nodes[_].kind == 1]):
			
			self.nodes[decision_node].denominator = 0
			self.nodes[decision_node].numerator = ([0 for _ in range(len(self.nodes[decision_node].primes))])
			
			for i in range((len(self.csv))):
				row = list(self.csv.iloc[i, :])
				if decision_node != self.root:
					feasible, branch = self.learning(decision_node, row)
					if feasible:
						self.nodes[decision_node].denominator += 1
						self.nodes[decision_node].numerator[branch] += 1
				else: # root node
					for k, branch in enumerate(self.nodes[decision_node].primes):
						mini_csdd = self.sub_csdd(branch)
						mini_csdd.logical_evidence = row
						if mini_csdd.logic_inference() == 1:
							self.nodes[decision_node].numerator[k] += 1
							break
				self.nodes[decision_node].denominator = sum(self.nodes[decision_node].numerator)
			
			# Logically impossible branches
			for branch, my_sub in enumerate(self.nodes[decision_node].subs):
				if self.nodes[my_sub].kind == 0:
					if self.nodes[my_sub].state == 2:
						assert self.nodes[decision_node].numerator[branch] == 0
						self.nodes[decision_node].numerator[branch] = -1

		# Terminal (top) nodes
		for top_node in [_ for _ in self.nodes if self.nodes[_].kind == 0]:
			if self.nodes[top_node].state == 3:
				
				self.nodes[top_node].numerator = 0
				self.nodes[top_node].denominator = 0
				
				for i in range((len(self.csv))):
					row = list(self.csv.iloc[i, :])
					feasible, branch = self.learning(top_node, row) # DEBUG: Branch unused?
					if feasible:
						self.nodes[top_node].denominator += 1
						if row[self.nodes[top_node].lit - 1]:
							self.nodes[top_node].numerator += 1
	
	def learn_credal_sets(self, ess = 1.0, eps = 0.01):	# DEBUG: Better management of epsilon
		
		for node in self.nodes:
			
			if self.nodes[node].kind == 1: # Decision nodes
				
				if len(self.nodes[node].thetas) > 1:
				
					zeros = False
					n = self.nodes[node].denominator
					
					if n == 0 and min(self.nodes[node].numerator) >= 0:  # Vacuous model when no data
						
						self.nodes[node].thetas = [[eps, 1 - eps] for _ in self.nodes[node].numerator]
					
					else:
						
						for aa, bb in enumerate(self.nodes[node].numerator):
							
							if bb == -1:
								
								self.nodes[node].thetas[aa] = [0, 0]
								zeros = True
							
							else:
								
								self.nodes[node].thetas[aa] = [(bb + eps) / (n + ess), (bb + ess - eps) / (n + ess)]
						
						if zeros and len(self.nodes[node].thetas) == 2:
							
							for k, theta in enumerate(self.nodes[node].thetas):
								
								if theta != [0, 0]:
									
									self.nodes[node].thetas[k] = [1, 1]
				
				else:
					self.nodes[node].thetas[0] = [1,1]
					
			elif self.nodes[node].state == 3: # Terminal (top) nodes
				
				m = self.nodes[node].numerator
				n = self.nodes[node].denominator
				
				if n == 0:
					self.nodes[node].theta = [eps, 1 - eps]
				else:
					self.nodes[node].theta = [(m + eps) / (n + ess), (m + ess - eps) / (n + ess)]
	
	def read_csdd(self,filename):		# Read CSDD file (check)
		with open(filename, "r") as text_file:
			a = text_file.read()
			for line in a.split("\n"):
				pieces = line.split(' ')
				if pieces[0]=='T':
					identifier = int(pieces[1])
					kind = int(pieces[2])
					lit = int(pieces[3])
					if kind != 3:
						self.nodes[identifier] = TerminalNode(identifier, lit, kind)
					else:
						th = [float(pieces[4]),float(pieces[5])]
						
						self.nodes[identifier] = TerminalNode(identifier, lit, kind, int(pieces[6]), th)
				
				if pieces[0]=='D':
					identifier = int(pieces[1])
					vtree_id = int(pieces[-1])
					b = int((len(pieces)-3)/4)
					children = pieces[2:-1]
					qq = [children[i:i+4] for i in range(0, len(children), 4)]
					primes = []
					subs = []
					thetas = []
					for ee in qq:
						primes.append(int(ee[0]))
						subs.append(int(ee[1]))
						thetas.append([float(ee[2]),float(ee[3])])
					self.nodes[identifier] = DecisionNode(identifier,vtree_id,primes,subs,thetas)
	
	def csdd2psdd(self):	# Compute a psdd from a csdd
		
		self.psdd = copy.deepcopy(self)
		self.psdd.opt = 0
		
		for node in self.nodes:
			
			if self.nodes[node].kind == 0:
				
				if self.nodes[node].state == 3:
					interval = self.nodes[node].theta
					self.psdd.nodes[node].theta = sum(interval) / 2.0
			
			if self.nodes[node].kind == 1:
				
				thetas2 = []
				
				for k, theta in enumerate(self.nodes[node].thetas):
					
					if k != len(self.nodes[node].thetas) - 1:
						
						thetas2.append(sum(theta) / 2.0)
					
					else:
						
						thetas2.append(sum(theta) / 2.0)
				
				thetas2[-1] = 1.0 - sum(thetas2[:-1])
				
				self.psdd.nodes[node].thetas = thetas2
	
	def set_evidence(self, e):
		
		#assert len(e) == len(self.nodes)
		self.evidence = e
	
	def set_optimum(self, o):
		
		self.opt = o
	
	def clean_messages(self):
		for node in self.nodes:
			self.nodes[node].message = -1.0
			self.nodes[node].logical_message = -1
	
	def compute(self, evidences, verbose=False, eps= 0.01):
		
		result = dict()
		queries = []
		observations = []
		explanations = []
		
		for (lit, evi) in zip(self.literals, evidences):
			if evi in [0, 1]:
				observations.append(str(lit) + '=' + str(evi))
			if evi in [2, 3]:
				queries.append(str(lit) + '=' + str(evi - 2))
			if evi == 4:
				explanations.append(str(lit))
		
		if explanations:
			assert not queries, 'No queries for MAP'
			# TODO: Add find_map
			# self.opt = 1
			if self.opt != 0:
				result[-1] = self.inference(-1, 2, evidences)
				result[1] = self.inference(1, 2, evidences)
				self.find_map(evidences.count(4))
				prob = ','.join(
					[str(a) + '=' + str(b) for a, b, c in zip(self.literals, self.xstar, self.evidence) if c == 4])
			else:
				result[-1] = self.inference(0, 2, evidences)
				result[1] = result[-1]
				self.find_map(evidences.count(4))
				prob = ','.join(
					[str(a) + '=' + str(b) for a, b, c in zip(self.literals, self.xstar, self.evidence) if c == 4])
			
			prob = 'P(' + prob
			if len(observations) > 0:
				prob += '|' + ','.join(observations)
			prob += ')*'
			
		elif not queries:
			
			if self.opt != 0:
				result[-1] = self.inference(-1, 0, evidences)
				result[+1] = self.inference(+1, 0, evidences)
			else:
				self.opt = -1
				result[-1] = self.inference(0, 0, evidences)
				result[+1] = result[-1]  # self.inference(0, 0, evidences)
			prob = 'P(' + ','.join(observations) + ')'
		
		else:
			assert len(queries) == 1
		
			optimization = [-1, +1]
		
			for o in optimization:  # Todo move bisection to a separate method

				a = 0.0000
				b = 1.0000
				fa = self.inference(o, 1, evidences, a)
				fb = self.inference(o, 1, evidences, b)
				
				assert fa * fb <= 0 , 'Cannot run bisection'
				if fa == 0 and fb == 0:
					#print(evidences)
					assert False, 'This should not happen' # c = -0.0001
				elif fa == 0 and fb < 0:
					c = 0.0
				elif fa == 0  and fb > 0:
					c = 0.0
				elif fa > 0 and fb ==0:
					c = 1.0
				elif fa < 0 and fb == 0:
					c = 1.0
				else:
					c = a -fa/(fb-fa)*(b-a)
					fc = self.inference(o, 1, evidences, c)
					if fa * fc < 0:
						b = c
					else:
						a = c
					#if verbose:
					#	print('f(a=%2.4f)=%2.7f,f(b=%2.4f)=%2.7f' % (a, fa, b, fb))
					while (b - a) > eps:  # DEBUG
						c = (a + b) / 2
						fc = self.inference(o, 1, evidences, c)
						#if verbose:
						#	print('f(a=%2.4f)=%2.7f,f(b=%2.4f)=%2.7f' % (a,fa, b,fb))
						if fa * fc < 0:
							b = c
						else:
							a = c
							fa = fc
				result[o] = c
			prob = 'P(' + queries[0] + '|' + ','.join(observations) + ')'
		if verbose:
			print('%2.8f <= %s <= %2.8f' % (result[-1], prob, result[1]))
		return result
	
	# TODO
	# self.psdd.inference()
	
	
	def sub_csdd(self, origin):
		
		go_ahead = True
		
		origin = [origin]
		
		while go_ahead:
			
			old_origin = origin
			
			for node in origin:
				
				if self.nodes[node].kind == 1:
					children = self.nodes[node].primes + self.nodes[node].subs
					origin = list(set(origin) | set(children))
			
			if len(origin) == len(old_origin):
				go_ahead = False
		
		csdd = Csdd()
		
		for k in origin:
			csdd.nodes[k] = self.nodes[k]
		
		csdd.clean_messages()
		
		return csdd
	
	# Inference methods
	
	
	def compute_logical_message_terminal(self, node):
		
		node.logical_message = 0
		
		if node.kind == 0:  # Terminal
			
			if node.state == 0:  # False
				
				if self.logical_evidence[node.lit - 1] == 0:  # Observed = True
					
					node.logical_message = 1
			
			if node.state == 1:  # True
				
				if self.logical_evidence[node.lit - 1] == 1:  # Observed = False
					
					node.logical_message = 1  # Inconsistent OR unobserved
			
			if node.state == 3:  # 'Top'
				
				node.logical_message = 1  # Unobserved
	
	def compute_message_terminal(self, node):
		
		if self.problem == 0:  # Marginal query
			
			node.message = 0.0  # Bot (node.state == 2)
			
			if node.state == 0:  # Literal is False
				node.message = 1.0  # Obs = False OR n/a
				if self.evidence[node.lit - 1] == 1:  # Obs = True
					node.message = 0.0
			
			if node.state == 1:  # Literal is True
				node.message = 1.0  # Obs = True OR n/a
				if self.evidence[node.lit - 1] == 0:  # Obs = False
					node.message = 0.0
			
			if node.state == 3:  # Top
				
				if self.evidence[node.lit - 1] == -1:  # Obs = n/a
					node.message = 1.0
				
				if self.evidence[node.lit - 1] == 1:  # Obs = True
					if self.opt == 0:  # P (PSDD)
						node.message = node.theta
					if self.opt == -1:  # lP (CSDD)
						node.message = node.theta[0]
					if self.opt == 1:  # uP (CSDD)
						node.message = node.theta[1]
				
				if self.evidence[node.lit - 1] == 0:  # Obs = False
					if self.opt == 0:  # P (PSDD)
						node.message = 1.0 - node.theta
					if self.opt == -1:  # lP (CSDD)
						node.message = 1.0 - node.theta[1]
					if self.opt == 1:  # uP (CSDD)
						node.message = 1.0 - node.theta[0]
		
		if self.problem == 1:  # Conditional query
			
			node.message = 0.0  # Bot (node.state == 2)
			
			if node.state == 0:  # Literal is False
				node.message = 0.0
				if self.evidence[node.lit - 1] == 2:  # Queried and False
					node.message = 1 - self.mu
				if self.evidence[node.lit - 1] == 3:  # Queried and True
					node.message = -self.mu
			
			if node.state == 1:  # Literal is True
				if self.evidence[node.lit - 1] == 2:  # Queried and False
					node.message = - self.mu
				if self.evidence[node.lit - 1] == 3:  # Queried and True
					node.message = 1 - self.mu
			
			if node.state == 3:  # Top
				
				if self.evidence[node.lit - 1] == 2:  # Queried and False
					if self.opt == -1:
						node.message = 1 - node.theta[1] - self.mu
					if self.opt == +1:
						node.message = 1 - node.theta[0] - self.mu
				
				if self.evidence[node.lit - 1] == 3:  # Queried and True
					if self.opt == -1:
						node.message = node.theta[0] - self.mu
					if self.opt == +1:
						node.message = node.theta[1] - self.mu

		
		if self.problem == 2:  # MAP query maximax for CSDD and map for PSDD
			
			node.message = 0.0
			node.m = 0.0
			
			
			
			if node.state in [0, 1]:  # Literals
				
				if self.evidence[node.lit - 1] == 4:  # node = unobserved literal
					
					node.message = 1.0  # A literal gives message one as it has the freedom to take the optimal state
					node.m = 1.0
					node.map = node.state  # MAP state is the literal state
				
				else:  # node = evidence
					
					if self.evidence[node.lit - 1] == node.state:  # node is an observed literal if consistent takes one
						
						node.message = 1.0
						node.m = 1.0
					
					else:
						
						node.message = 0.0
						node.m = 0.0
			
			
			elif node.state == 3:  # Top
				
				if self.evidence[node.lit - 1] == 4:
					
					if self.opt == 0:  # PSDD
						
						if node.theta > 0.5:  # p > 1-p
							node.message = node.theta
							node.m = node.theta
							node.map = 1
						else:
							node.message = 1.0 - node.theta
							node.m = 1.0 - node.theta
							node.map = 0
					else:  # CSDD max only
						if node.theta[1] > (1 - node.theta[0]):
							node.message = node.theta[1]
							node.m = node.theta[1]
							node.map = 1
						else:
							node.message = 1 - node.theta[0]  # 1-l
							node.m = 1 - node.theta[0]  # 1-l
							node.map = 0
				else:  # if the node is associated to an observed literal
					
					if self.opt == 0:  # PSDD
						
						node.message = 1 - node.theta
						
						if self.evidence[node.lit - 1] == 1:
							node.message = node.theta
					
					elif self.opt == 1:
						
						node.message = 1 - node.theta[0]
						node.m = 1 - node.theta[0]
						
						if self.evidence[node.lit - 1] == 1:
							node.message = node.theta[1]
							node.m = node.theta[1]
					
					else:  # Minimizing
						
						node.message = 1 - node.theta[1]
						node.m = 1 - node.theta[1]
						
						if self.evidence[node.lit - 1] == 1:
							node.message = node.theta[0]
							node.m = node.theta[0]
		
		if self.problem == 3:  # Robustness CSDD query
			
			
			if node.state < 2:  # Literal
				
				if self.evidence[node.lit - 1] == 4:  # To be explained
				
					node.message = 1.0
					#if node.state == self.xstar[node.lit - 1]:
					#	node.message = 0.0
				
				else:  # Observed
				
					node.message = 0.0

					if node.state == self.evidence[node.lit - 1]:
						assert self.psdd.xstar[node.lit - 1] == node.state
						node.message = 1.0

			elif node.state == 3:  # Top
				if self.psdd.xstar[node.lit - 1]:
					node.message = max(1,(1-node.theta[0])/node.theta[0])
				else:
					node.message = max(1,node.theta[1]/(1-node.theta[1]))

			elif node.state == 4:  # Bot
				node.message = 0  # DEBUG: forse e' 1?

	
	def compute_logical_message_decision(self, node):
		
		node.logical_message = 0
		
		for (prime, sub) in zip(node.primes, node.subs):
			
			if self.nodes[prime].logical_message * self.nodes[sub].logical_message == 1:
				node.logical_message = 1
				
				break
	
	def compute_message_decision(self, node):
		
		
		if self.opt == 0:  # P (PSDD)
			
			if self.problem == 0:  # Marginal query
				
				node.message = 0.0
				
				for (theta, prime, sub) in zip(node.thetas, node.primes, node.subs):
					assert self.nodes[prime] != -1.0, 'Bad var order'
					assert self.nodes[sub] != -1.0, 'Bad var order'
					node.message += theta * self.nodes[prime].message * self.nodes[sub].message
			
			if self.problem == 2:  # Credal MAP query # this is an argmax
				
				candidates = []
				
				for (theta, prime, sub) in zip(node.thetas, node.primes, node.subs):
					assert self.nodes[prime] != -1.0, 'Bad var order'
					assert self.nodes[sub] != -1.0, 'Bad var order'
					candidates.append(theta * self.nodes[prime].message * self.nodes[sub].message)
				
				node.message = max(candidates)
				node.map = candidates.index(node.message)
		
		else:  # CSDD
			
			if self.problem in [0, 1]:
				
				all_zeros = True
				x_vars = []
				objective = []
				
				for i in range(len(node.thetas)):
					x_vars.append(LpVariable('x' + str(i), lowBound=node.thetas[i][0], upBound=node.thetas[i][1],
											 cat='Continuous'))
				
				for x, prime, sub in zip(x_vars, node.primes, node.subs):
					
					if self.problem == 0:  # Marginal
						
						coefficient = self.nodes[prime].message * self.nodes[sub].message
					
					if self.problem == 1:  # Conditional
						
						# if node.id == 173:
						#	print('Eccomi',prime,self.nodes[prime].message,sub,self.nodes[sub].message)
						
						assert len([e for e in self.evidence if e >= 2]) == 1, 'Only single queries'
						
						queried = [i for i, e in enumerate(self.evidence) if e >= 2][0]
						
						
						if (queried + 1) in self.nodes[node.id].left:
							
							message1 = self.nodes[prime].message
							
							r_csdd = self.sub_csdd(sub)
							
							if self.opt == -1:
								if message1 < 0:
									message2 = r_csdd.inference(+1, 0, self.evidence)  # sub evi
								else:
									message2 = r_csdd.inference(-1, 0, self.evidence)
							
							if self.opt == +1:
								if message1 < 0:
									message2 = r_csdd.inference(-1, 0, self.evidence)
								else:
									message2 = r_csdd.inference(+1, 0, self.evidence)
						
						elif (queried + 1) in self.nodes[node.id].right:
							
							message2 = self.nodes[sub].message
							
							l_csdd = self.sub_csdd(prime)
							
							if self.opt == -1:
								if message2 < 0:
									message1 = l_csdd.inference(+1, 0, self.evidence)
								else:
									message1 = l_csdd.inference(-1, 0, self.evidence)
							if self.opt == +1:
								if message2 < 0:
									message1 = l_csdd.inference(-1, 0, self.evidence)
								else:
									message1 = l_csdd.inference(+1, 0, self.evidence)
						else:
							message1 = 0
							message2 = 0
							
						coefficient = message1 * message2
					
					if coefficient != 0:  # Check if this is the case
						objective.insert(-1, x * coefficient)
						all_zeros = False
				
				if all_zeros:
					node.message = 0
				else:
					if self.opt == -1:  # lP (CSDD)
						my_model = LpProblem("Minimizing", LpMinimize)
					if self.opt == 1:  # uP (CSDD)
						my_model = LpProblem("Maximizing", LpMaximize)
					my_model += lpSum(objective)
					my_model += (pulp.lpSum(x_vars) == 1)
					my_model.solve()
					node.message = value(my_model.objective)
			
			elif self.problem == 2:  # Credal MAP
				candidate_messages = []
				for (theta, prime, sub) in zip(node.thetas, node.primes, node.subs):
					assert self.nodes[prime] != -1.0, 'Bad var order'
					assert self.nodes[sub] != -1.0, 'Bad var order'
					#print(theta,'errrr')
					candidate_messages.append(theta[1] * self.nodes[prime].message * self.nodes[sub].message)
				node.message = max(candidate_messages)
				node.m = max(candidate_messages)
				# TODO: remove it?
				node.map = candidate_messages.index(node.message)
			
			elif self.problem == 3:  # Robustness in decision nodes
				
				if node.red:
				
					# Finding the "j" branch
					for kkk, ooo in enumerate(node.primes):
						oo_csdd = self.sub_csdd(ooo)
						oo_csdd.logical_evidence = self.psdd.xstar
						if oo_csdd.logic_inference():
							j = kkk


				
					# Preparing the sub-CSDD to be used in the denominator
					ll_csdd = self.sub_csdd(node.primes[j])
					rr_csdd = self.sub_csdd(node.subs[j])


					aa_csdd = copy.deepcopy(ll_csdd)
					bb_csdd = copy.deepcopy(rr_csdd)
					
					# Marginal lower probabilities of the sub-evidence in the sub-CSDD
					message1 = aa_csdd.inference(-1, 0, self.psdd.xstar)
					message2 = bb_csdd.inference(-1, 0, self.psdd.xstar)

					
					assert message1 > 0, 'Lilith s conjecture'
					
					denominator = message1 * message2
					
					possible_messages = []
					i = 0
					for (prime, sub) in zip(node.primes, node.subs):
						if i != j:
							
							lll_csdd = self.sub_csdd(node.primes[i])
							rrr_csdd = self.sub_csdd(node.subs[i])
							aaa_csdd = copy.deepcopy(lll_csdd)
							bbb_csdd = copy.deepcopy(rrr_csdd)
							
							coefficient2 = aaa_csdd.inference(1, 2, self.evidence) * bbb_csdd.inference(1, 2, self.evidence)
							
							possible_messages.append(coefficient2 / denominator * node.thetas[i][1] / node.thetas[j][0])
						else:
							possible_messages.append(self.nodes[node.primes[j]].message * self.nodes[node.subs[j]].message)
						i += 1
					node.message = max(possible_messages)
					#print(possible_messages)
				else:
					node.message = 0
	def find_map(self, n):
		
		processed = []  # Nodes already processed
		
		assert len(self.nodes) > 1, 'To add MAP for single node CSDD'
		
		assert self.root >= 0, 'bad root'
		
		self.xstar = copy.deepcopy(self.evidence)
		
		for node in self.nodes:
			self.nodes[node].visited = 0
		
		active_node = self.root
		assert self.nodes[active_node].kind == 1
		self.nodes[active_node].visited = 1
		active_branch = self.nodes[active_node].map
		
		while len(processed) != n:
			if self.nodes[active_node].kind == 0:  # If terminal
				if self.evidence[a_node.lit - 1] == 4:  # DEBUG
					processed.append((a_node.lit, a_node.map))
					self.xstar[a_node.lit - 1] = a_node.map
				active_node = self.nodes[active_node].parent
			else:  # If the active node is a decision node
				active_branch = self.nodes[active_node].map
				if self.nodes[
					self.nodes[active_node].primes[active_branch]].visited == 0:  # and its active prime is unvisited
					a_node = self.nodes[self.nodes[active_node].primes[active_branch]]  # Let's visit the active prime
					a_node.visited = 1
					a_node.parent = active_node  # not forgetting to notice the parent
					active_node = a_node.id
					active_branch = a_node.map
				
				elif self.nodes[
					self.nodes[active_node].subs[active_branch]].visited == 0:  # and its active prime is unvisited
					a_node = self.nodes[self.nodes[active_node].subs[active_branch]]  # Let's visit the active prime
					a_node.visited = 1
					a_node.parent = active_node  # not forgetting to notice the parent
					active_node = a_node.id
					active_branch = a_node.map
				else:
					# if active_node != self.root:
					active_node = self.nodes[active_node].parent
		return processed
	
	def paint_red(self, verbose = False):
		
		for my_node in self.nodes:
			self.nodes[my_node].red = False
		
		
		self.nodes[self.root].red = True
		
		# Starting from the root node
		active_node = self.root
		root_activated = 1

		while root_activated < 3:
			# Check the prime children
			for idx, prime in enumerate(self.nodes[active_node].primes):
				
				oo_csdd = self.sub_csdd(prime)
				oo_csdd.logical_evidence = self.psdd.xstar
				
				# Let us find the consistent prime
				if oo_csdd.logic_inference():
					
					# The consistent prime can be a decision node or a terminal
					# if it is a terminal there is nothing to do (no else)
					if self.nodes[prime].kind == 1:
						
						# If the consistent prime is not red (we already know it is a dec) we paint it
						if not self.nodes[prime].red:
							self.nodes[prime].red = True
							# and we set as active the node
							active_node = prime
							if verbose:
								print("Active Node:", active_node,'Down Left')
						
						# if the consistent prime was already red, but its sub not we paint
						# and activate the sub (in case it is a decision)
						elif self.nodes[self.nodes[active_node].subs[idx]].kind == 1:
							
							if not self.nodes[self.nodes[active_node].subs[idx]].red:
								self.nodes[self.nodes[active_node].subs[idx]].red = True
								active_node = self.nodes[active_node].subs[idx]
								if verbose:
									print("Active Node:", active_node, 'Down Right')
							else:
								for node_red in self.nodes:
									if self.nodes[node_red].kind == 1:
										if self.nodes[node_red].red:
											children = self.nodes[node_red].primes + self.nodes[node_red].subs
											if active_node in children:
												active_node = node_red
												if verbose:
													print("Active Node:", active_node, 'Up')
												break
						else:
							for node_red in self.nodes:
								if self.nodes[node_red].kind == 1:
									if self.nodes[node_red].red:
										children = self.nodes[node_red].primes+self.nodes[node_red].subs
										if active_node in children:
											active_node = node_red
											if verbose:
												print("Active Node:", active_node, 'Up')
											break
					# no else?
					elif self.nodes[self.nodes[active_node].subs[idx]].kind == 1:
						if not self.nodes[self.nodes[active_node].subs[idx]].red:
							self.nodes[self.nodes[active_node].subs[idx]].red = True
							active_node = self.nodes[active_node].subs[idx]
							if verbose:
								print("Active Node:", active_node, 'Down Right')
						else:
							for node_red in self.nodes:
								if self.nodes[node_red].kind == 1:
									if self.nodes[node_red].red:
										children = self.nodes[node_red].primes+self.nodes[node_red].subs
										if active_node in children:
											active_node = node_red
											if verbose:
												print("Active Node:", active_node, 'Up')
											break
							
					else:
						for node_red in self.nodes:
							if self.nodes[node_red].kind == 1:
								if self.nodes[node_red].red:
									children = self.nodes[node_red].primes+self.nodes[node_red].subs
									if active_node in children:
										active_node = node_red
										if verbose:
											print("Active Node:", active_node, 'Up')
										break
				
				if active_node == self.root:
					root_activated += 1
					if verbose:
						print("Root, again")
		
	def inference(self, opt, prob, evi, mu=-1.0):
		
		self.mu = mu
		self.opt = opt
		self.problem = prob
		self.set_evidence(evi)
		to_process = []
		# Preprocessing / cleaning
		if self.problem == 3:
			for node in self.nodes:
				self.nodes[node].map = self.psdd.nodes[node].map
		for id_node, this_node in self.nodes.items():
			if this_node.kind == 0:  # TERMINAL
				self.compute_message_terminal(this_node)
			else:  # DECISION
				to_process.append(id_node)
		if len(self.nodes) == 1:
			for node in self.nodes:
				root = node
		else:
			while len(to_process) > 0:
				#print('TO PROC',len(to_process))
				if len(to_process) == 1:
					root = to_process[0]
					#print('root is',root,to_process)
				for node2 in to_process:
					#print('processing',node2)
					children = self.nodes[node2].primes + self.nodes[node2].subs
					#print(children)
					#print([v for v in children if v in to_process])
					#print('---')
					if not len([v for v in children if v in to_process]):
						self.compute_message_decision(self.nodes[node2])
						to_process.remove(node2)
						break
					#else:
					#	break
		self.root = root  # DEBUG: Finding the root of the csdd?
		
		return self.nodes[root].message
	
	def logic_inference(self):
		
		to_process = []
		
		for id_node, this_node in self.nodes.items():
			
			if this_node.kind == 0:  # TERMINAL
				
				self.compute_logical_message_terminal(this_node)
			
			else:  # DECISION
				
				to_process.append(id_node)
		
		if len(self.nodes) == 1:
			
			# TODO: Verificare, non serve il for
			for node in self.nodes:
				
				root = node
		
		else:
			
			while to_process:
				
				if len(to_process) == 1:
					root = to_process[0]
				
				for node2 in to_process:
					
					children = self.nodes[node2].primes + self.nodes[node2].subs
					
					if not len([v for v in children if v in to_process]):
						self.compute_logical_message_decision(self.nodes[node2])
						to_process.remove(node2)
						break
		
		self.root = root  # Finding the root of the csdd
		
		return self.nodes[root].logical_message
	
	def find_paths(self, node):  # Compute the path (i.e., list of primes) from the node 'node' to the root
		paths = []
		visited = []
		proseguire = True
		while proseguire:
			active_node = node
			path = []
			while active_node != self.root:  # or len(q) != len(self.nodes):
				for node2 in self.nodes:
					if self.nodes[node2].kind == 1:
						if active_node in self.nodes[node2].primes and node2 not in visited:
							pos = self.nodes[node2].primes.index(active_node)
							mysub = self.nodes[node2].subs[pos]
							path.append((active_node, mysub, 'prime'))
							visited.append(active_node)
							active_node = node2
							break
						elif active_node in self.nodes[node2].subs and node2 not in visited:
							pos = self.nodes[node2].subs.index(active_node)
							myprime = self.nodes[node2].primes[pos]
							path.append((myprime, active_node, 'sub'))
							visited.append(active_node)
							active_node = node2
							break
				else:
					proseguire = False
					break
			path.append((active_node, 'root'))
			paths.append(path)
			if len(paths) == 2 and len(path) == 2:  # DEBUG: better
				break
			active_node = node
		return paths[:-1]
	
	def learning(self, decision_node, log_evi):
		self.logical_evidence = log_evi
		feasible = -2
		branch = -2
		for path in self.find_paths(decision_node):
			feasible = 0
			for step in path:
				if step[-1] == 'sub' or step[-1] == 'prime':  # If sub, compute the message on the prime
					minicsdd = self.sub_csdd(step[0])
					minicsdd.logical_evidence = self.logical_evidence
					if minicsdd.logic_inference() == 0:
						break
					else:
						continue  # TODO: CHECK
				else:
					feasible = 1
			branch = -1
			if feasible and self.nodes[decision_node].kind == 1:
				for k, prime2 in enumerate(self.nodes[decision_node].primes):
					minicsdd2 = self.sub_csdd(prime2)
					minicsdd2.logical_evidence = self.logical_evidence
					if minicsdd2.logic_inference() == 1:
						branch = k
						break
				break
		return feasible, branch
	

	
	def print_log(self, verbose = False):
		for _, node in sorted(self.nodes.items()):
			if node.kind == 0:
				node_description = '[T%d] ' % node.id
				if node.state == 0:
					node_description += '[/%s]' % self.literals[node.lit - 1]
				if node.state == 1:
					node_description += '[%s]' % self.literals[node.lit - 1]
				if node.state == 2:
					node_description += '[BOT|%s]' % self.literals[node.lit - 1]
				if node.state == 3:
					if self.opt != 0:
						node_description += '[TOP|%s:%1.4f-%1.4f]' % (
							self.literals[node.lit - 1], node.theta[0], node.theta[1])
					else:
						node_description += '[TOP|%s:%1.4f]' % (self.literals[node.lit - 1], node.theta)
				node_description += '(%d)' % ((node.lit - 1) * 2)
				self.serial += node_description + '\n'
			else:
				node_description = '[D%d]' % node.id
				if node.id == self.root:
					node_description += '*'
				for (p, s, t) in zip(node.primes, node.subs, node.thetas):
					if self.nodes[p].kind == 0 and self.nodes[s].kind == 0:
						if self.opt != 0:
							node_description += "[T%d|T%d]:%1.4f-%1.4f" % (p, s, t[0], t[1])
						else:
							node_description += "[T%d|T%d]:%1.4f" % (p, s, t)
					if self.nodes[p].kind == 0 and self.nodes[s].kind == 1:
						if self.opt != 0:
							node_description += "[T%d|D%d]:%1.4f-%1.4f" % (p, s, t[0], t[1])
						else:
							node_description += "[T%d|D%d]:%1.4f" % (p, s, t)
					if self.nodes[p].kind == 1 and self.nodes[s].kind == 0:
						if self.opt != 0:
							node_description += "[D%d|T%d]:%1.4f-%1.4f" % (p, s, t[0], t[1])
						else:
							node_description += "[D%d|T%d]:%1.4f" % (p, s, t)
					if self.nodes[p].kind == 1 and self.nodes[s].kind == 1:
						if self.opt != 0:
							node_description += "[D%d|D%d]:%1.4f-%1.4f" % (p, s, t[0], t[1])
						else:
							node_description += "[D%d|D%d]:%1.4f" % (p, s, t)
				node_description += '(%d)' % node.vtree
				self.serial += node_description + '\n'
		if verbose:
			print(self.serial)
	
	
	def write_csdd(self, filename):
		serial = ''
		for _, node in sorted(self.nodes.items()):
			if node.kind == 0:
				node_description = 'T %d ' % node.id
				if node.state == 0:
					node_description += '0 %d ' % node.lit
				if node.state == 1:
					node_description += '1 %d ' % node.lit
				if node.state == 2:
					node_description += '2 %d ' % node.lit
				if node.state == 3:
					node_description += '3 %d %1.4f %1.4f %d' % (
						node.lit, node.theta[0], node.theta[1], (node.lit-1)*2, )
				serial += node_description + '\n'
			else:
				node_description = 'D %d' % node.id
				for (p, s, t) in zip(node.primes, node.subs, node.thetas):
					node_description += " %d %d %1.4f %1.4f" % (p, s, t[0], t[1])
				node_description += ' %d' % node.vtree
				serial += node_description + '\n'
		with open(filename, "w") as text_file:
			print(serial, file=text_file)
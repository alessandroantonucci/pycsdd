EPS = 0.0001

class Node:
	def __init__(self, identifier):
		self.id = identifier
		self.kind = -1          # 0 = terminal / 1 = decision
		self.message = -1.0


class TerminalNode(Node):

	def __init__(self, identifier=-1, variable=-1,  stt=-1, vtreev=-1, t=-1):
		self.id = identifier        # id
		self.kind = 0               # 0 for terminal
		self.lit = variable         # literal in the node
		self.vtree = vtreev         # vtree node associated
		self.state = stt            # False/True/Bot/Top F/T = variable is ...
		if stt==3:
			if t==-1:
				self.theta = [EPS, 1 - EPS]        # probability (for top only)
			else:
				self.theta = t
		self.message = -1.0         # messages (to be computed)
		self.logical_message = -1   #
		self.map = -1                # Most probable state
		self.visited = 0
		self.parent = -1
		self.m = -1.0               # Message computed for maximax
		self.logical_message = -1
		self.numerator = -1
		self.denominator = -1

class DecisionNode(Node):

	def __init__(self, identifier, vtree, p, s, t=-1):
		self.id = identifier        # id
		self.kind = 1               # 1 for decision
		self.primes = p         	# ids of the primes
		self.subs = s             	# ids of the subs
		self.vtree = vtree          # vtree node associated
		self.message = -1.0         # messages (to be computed)
		self.logical_message = -1   #
		self.left = []              # left scope
		self.right = []             # right scope
		self.map = -1               # Most probable branch
		self.visited = 0
		self.parent = -1
		self.active_branch = -1
		self.m = -1.0               # Message computed for maximax
		self.numerators = []
		self.denominator = -1
		self.red = False
		if t==-1:
			self.thetas = [[EPS, 1.0-EPS] for _ in range(len(p))]
		else:
			self.thetas = t
		# probabilities

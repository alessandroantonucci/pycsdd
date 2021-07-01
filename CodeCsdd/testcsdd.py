from csdd import *
import unittest
import warnings
warnings.filterwarnings("ignore")

VERBOSE = True

class TestCsdd(unittest.TestCase):

	def test_csdd_singly(self):

		if True:

			credal = Csdd()
			credal.set_csv('../Data/singly.csv')
			credal.set_vtree('../Data/singly.vtree')
			credal.set_sdd('../Data/singly.sdd')
			credal.set_root()
			credal.set_left_right()
			credal.learn_counts()
			credal.learn_credal_sets(1.0)
			credal.print_log(VERBOSE)

			# # Joint queries for the 16 possible universes
			# for i in range(2**4):
			# 	sequence = [int(x) for x in list('{0:0b}'.format(i))]
			# 	while len(sequence)<4:
			# 		sequence.insert(0,0)
			# 	credal.compute(sequence,VERBOSE)
			#
			# # Marginal queries
			# for i in range(4):
			# 	sequence = [-1 for _ in range(4)]
			# 	sequence[i] = 1
			# 	credal.compute(sequence, VERBOSE)
			#
			# # Conditional queries
			# credal.compute([3, -1, -1, 1], VERBOSE)      # P(L=1|A=1)
			# credal.compute([-1, 0, -1, 2], VERBOSE)      # P(A=0|K=0)
			
			# To compute a PSDD consistent with the CSDD
			credal.csdd2psdd()
			#credal.psdd.print_log(VERBOSE)
			# Compute the most probable configuration of the first three variables given the last variable true in the PSDD
			credal.psdd.compute([4, 4, 4, 4], VERBOSE)
			credal.xstar = credal.psdd.xstar
			credal.paint_red()
			credal.inference(1, 3, [4,4,4,4])


	def test_csdd_multi(self):

		if False:

			credal = Csdd()
			credal.set_csv('../Data/multi.csv')
			credal.set_vtree('../Data/multi.vtree')
			credal.set_sdd('../Data/multi.sdd')
			credal.set_root()
			credal.set_left_right()
			credal.learn_counts()
			credal.learn_credal_sets(2.0)
			credal.print_log(VERBOSE)
			
			# Full Joint queries
			for i in range(2 ** 4):
				sequence = [int(x) for x in list('{0:0b}'.format(i))]
				while len(sequence) < 4:
					sequence.insert(0, 0)
				credal.compute(sequence, VERBOSE)
			
			# Marginal queries
			for i in range(4):
				sequence = [-1 for _ in range(4)]
				sequence[i] = 1
				credal.compute(sequence, VERBOSE)
			
			# Conditional queries
			credal.compute([3, -1, -1, 1], VERBOSE)  # P(A=1|D=1)
			credal.compute([1, 1, 2, 0], VERBOSE)  # P(C=1|A=1,B=1,D=0)
			
			# To compute a PSDD consistent with the CSDD
			credal.csdd2psdd()
			credal.psdd.print_log(VERBOSE)
			# Compute the most probable configuration of the first three variables given the last variable true in the PSDD
			credal.psdd.compute([4, 4, 4, 4], VERBOSE)
			print(credal.psdd.xstar)
			credal.xstar = credal.psdd.xstar
			credal.paint_red()
			credal.inference(1, 3, [4, 4, 4, 4])
	

if __name__ == '__main__':
    unittest.main()
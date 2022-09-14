from .algo_ot import (
	OPW,
	TOPW1,
	TOPW2
)
import os
import sys
from pathlib import Path

dir_path = str(Path(os.path.dirname(os.path.realpath(__file__))).parent)
sys.path.append(dir_path)


from robustOPW import get_prob


class TrendOTDis:
	def __init__(self, lambda1, lambda2, delta, trend_method='l1', algo='opw'):
		if algo == 'opw':
			self.ot_solver = OPW(lambda1, lambda2, delta)
		elif algo == 'topw1':
			self.ot_solver = TOPW1(lambda1, lambda2, delta)
		elif algo == 'topw2':
			self.ot_solver = TOPW2(lambda1, lambda2, delta)
		else:
			raise NotImplementedError
		self.trend_method = trend_method

	def dist(self, x1, x2):
		x1_prob, x2_prob = get_prob(x1, x2)
		self.ot_solver.fit(x1, x2, x1_prob, x2_prob)
		return self.ot_solver.get_distance()

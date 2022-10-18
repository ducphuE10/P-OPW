from .algo_ot import (
	OPW,
	TOPW1,
	TOPW2
)
import os
import sys
from pathlib import Path

# dir_path = str(Path(os.path.dirname(os.path.realpath(__file__))).parent)
# sys.path.append(dir_path)


from robustOPW import get_prob, L1TrendFilter, RobustTrendFilter


class TrendOTDis:
	def __init__(self, lambda1, lambda2, delta, trend_method='l1', algo='opw', trend_args=None):

		if algo == 'opw':
			self.ot_solver = OPW(lambda1, lambda2, delta)
		elif algo == 'topw1':
			self.ot_solver = TOPW1(lambda1, lambda2, delta)
		elif algo == 'topw2':
			self.ot_solver = TOPW2(lambda1, lambda2, delta)
		else:
			raise NotImplementedError

		if trend_method == 'l1':
			self.trend_filter = L1TrendFilter(lambda1=trend_args.trend_lambda1, lambda2=trend_args.trend_lambda2)
		elif trend_method == 'robust':
			self.trend_filter = RobustTrendFilter(
				lambda1=trend_args.trend_lambda1,
				lambda2=trend_args.trend_lambda2,
				penalty=trend_args.trend_penalty,
				max_iter=trend_args.trend_max_iter,
			)
		else:
			raise NotImplementedError


	def dist(self, x1, x2, method='trend'):
		x1_prob, x2_prob = self.trend_filter.fit(x1, x2)
		x1_trend, x2_trend = None, None
		if method == 'trend':
			x1_trend, x2_trend = self.trend_filter.get_trend()
		self.ot_solver.fit(x1=x1, x2=x2, a=x1_prob, b=x2_prob, x1_trend=x1_trend, x2_trend=x2_trend)
		return self.ot_solver.get_distance()

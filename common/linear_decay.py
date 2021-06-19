import torch
import numpy as np


# linear decay for epsilon
class LinearDecay:
	def __init__(self, start, end, max_steps):
		self.start = start
		self.end = end
		self.max_steps = max_steps
		self.curr_steps = 0

		if max_steps > 0:
			self.delta = (start - end) / max_steps
			self.val = start
		else:
			self.val = end

	def step(self):
		if self.max_steps > 0:
			if self.start > self.end:
				self.val = max(self.val - self.delta, self.end)
			else:
				self.val = min(self.val - self.delta, self.end)

	def __call__(self):
		# if max_steps <= 0, self.val is always = start
		return self.val

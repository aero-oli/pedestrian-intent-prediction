# Implementation of a base model class

import numpy as np
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.module):
    	"""""
    	Base class implementation for all the models.
	The class is inherited from nn.Module
    	"""""
    	def __str__(self):
		"""
		Method to get the string representation of the object.

		Parameters
		----------
		self	: BaseModel
			  Instance of the class

		Returns
		-------
		string	: str
			  String representation of the object
		"""
		modelParameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([np.prod(p.size) for p in modelParameters])

		return super().__str__() + "\nTrainable Parameters: {}".format(params)

	@abstractmethod
	def forward(self, *inputs):
		"""
		Method to perform a forward pass on the neural network.
		The method is abstract. Therefore, the function must be implemented in the subclass.

		Parameters
		----------
		self	: BaseModel
			  Instance of the class
		*inputs	: Multiple
			  Variable number of non-keyword arguments

		Returns
		-------
		None
		"""
		raise NotImplementedError

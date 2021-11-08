import typing


BatchSize = typing.NewType("BatchSize", int)
NFeatures = typing.NewType("NFeatures", int)
NNeuronsCurr = typing.NewType("NumberNeuronsCurrentLayer", int)
NNeuronsPrev = typing.NewType("NumberNeuronsPreviousLayer", int)
NNeuronsOut = typing.NewType("NumberNeuronsOutput", int)

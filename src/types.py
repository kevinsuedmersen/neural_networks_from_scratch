import typing


NSamples = typing.NewType("NSamples", int)
BatchSize = typing.NewType("BatchSize", int)
NFeatures = typing.NewType("NFeatures", int)
NNeurons = typing.NewType("NumberNeuronsCurrentLayer", int)
NNeuronsPrev = typing.NewType("NumberNeuronsPreviousLayer", int)
NNeuronsNext = typing.NewType("NumberNeuronsNextLayer", int)
NNeuronsOut = typing.NewType("NumberNeuronsOutput", int)
ImgHeight = typing.NewType("ImageHeight", int)
ImgWidth = typing.NewType("ImageWidth", int)
ImgChannels = typing.NewType("ImageChannels", int)

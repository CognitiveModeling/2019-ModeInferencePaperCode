# Learning, Planning, and Control in a Monolithic Neural Event Inference Architecture
This repository includes the code for the paper: Learning, Planning, and Control in a Monolithic Neural Event Inference Architecture. Authors: Martin V. Butz, David Bilkey, Dania Humaidan, Alistair Knott and Sebastian Otte.

The project is structured as following:

The main class is RNNEval_CIEBdetection_MultiProblem.java, where an instance of the class ANN3InputComplexNet.java is created.
The type of the used complex network is LSTM: ANNLayer_LSTM.java that implements the interface ANNLayer.java.

The used simulator is RB3Simulator.java that implements CSCProblemAndOutInMapInterface.java which in turn extends ContinuousSequentialControlProblem.java.

The required activation functions and other tools are located in de/cogmod/utilities.

In case of questions, please reach Dania Humaidan at dania.humaidan@uni-tuebingen.de

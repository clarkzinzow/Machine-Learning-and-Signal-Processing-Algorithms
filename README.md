# Machine-Learning-and-Signal-Processing-Algorithms
MATLAB implementations of a variety of machine learning/signal processing algorithms.

---

This repository contains MATLAB implementations of a variety of popular machine learning algorithms, most of which were part of the graduate course in advanced machine learning (CS 761) at UW-Madison in the Spring of 2016.

List of algorithms implemented:

1. proximal gradient method
2. stochastic gradient descent
3. backpropagation
4. low-rank matrix reconstruction from partial sampling

All of the algorithms are heavily commented (possibly to a fault), but I wanted someone in the midst of a machine learning class to be able to read through the code and understand it decently well.  Although I have done my best to implement these algorithms with efficiency in mind (within the confines of MATLAB's inherent deficiencies in this regard), this repository is far more valuable as a teaching tool than a performance-centric library.

Due to the algorithms being so heavily commented, many implementation details are contained within the code as comments instead of in a README.

In the near future, I will include a demo folder that demonstrates the correctness and performance of each algorithm on a set of representative problems.  I also might create a README with implementation details for each algorithm, to be located in the src folder.
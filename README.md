Introduction
-------
This package is a Matlab implementation of the algorithms described in the classical machine learning textbook:
Pattern Recognition and Machine Learning by C. Bishop ([PRML](http://research.microsoft.com/en-us/um/people/cmbishop/prml/)).

Description
-------
The design goal of the code are as follows:

1. Clean: Code is very succinct. There are little nasty guarding code that distracts readers' attention. As a result, the core of the algorithms can be easily spot.
2. Efficient: Many tricks for making Matlab scripts efficient were applied (eg. vectorization and matrix factorization). Many functions are even comparable with C implementation. Usually, functions in this package are orders faster than Matlab builtin functions which provide the same functionality (eg. kmeans). If anyone found any Matlab implementation that is faster than mine, I am happy to further optimize.
3. Robust: Many numerical stability techniques are applied, such as probability computation in log scale to avoid numerical underflow and overflow, square root form update of symmetric matrix, etc.
4. Easy to learn: The code is heavily commented. Reference formulas in PRML book are indicated for corresponding code lines.
5. Practical: The package is designed not only to be easily read, but also to be easily used to facilitate ML research. Many functions in this package are already widely used  (see [Matlab file exchange](http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A49739)).


Installation
-------
1. Download the package by running: `git clone https://github.com/PRML/PRMLT.git`.

2. Run Matlab and navigate to package location as working directory, then run the init.m script.

3. Run some demos in the demo directory. Enjoy!

License
-------
Currently Released Under GPLv3


Contact
-------
sth4nth at gmail dot com

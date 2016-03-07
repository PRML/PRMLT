Pattern Recognition and Machine Learning
===========

This package is a matlab implementation of the algorithms described in the book:
Pattern Recognition and Machine Learning by C. Bishop ([PRML](http://research.microsoft.com/en-us/um/people/cmbishop/prml/))

The goal of the code are as follows:

1. Clean: making the code as succinct as possible, which means, there are little nasty guarding code that distracts reader's attention so that the core of the algorithms can be easily spot.
2. Efficient: utilizing matlab vectorization trick as much as possible to make the function fast. Many functions are even comparable with C implementation. Usually, functions in this package are orders faster than matlab builtin functions which provide same functionality (such as kmeans). If anyone found any matlab implementation which is faster than mine, I am happy to further optimize.
3. Robust: many numerical stability techniques are applied, such as probability computation in log scale to avoid numerical underflow and overflow, square root form update of symetric matrix, etc.
4. Easy to learn: the code is heavily commented. Reference formulas in PRML book are indicated for corresponding code lines.
5. Practical: the package is designed not only for users to learn the algorithms in the book, but also to facilitate ML research. Many functions in this package are already among the top downloads in Matlab [file exchange](http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A49739) site and widely used.

License
-------
Currently Released Under GPLv3


Contact
-------
sth4nth at gmail dot com


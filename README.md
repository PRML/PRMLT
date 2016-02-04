Pattern Recognition and Machine Learning
===========

This package contains the matlab implementation of the algorithms described in the book:
Pattern Recognition and Machine Learning by C. Bishop (http://research.microsoft.com/en-us/um/people/cmbishop/prml/)

The goal of the code are as follows:

1. clean. make the code as clean as possible, which means, there are little nasty guarding code that distracts reader's attention so that the core of the algorithms is easy to spot.
2. efficient. use matlab vectorization trick as much as possible to make the function fast, many functions are even comparable with c implementation. usually, the functions in this package are orders faster than matlab builtin function which provide same functionality (such as kmeans). If anyone can find any matlab implementation that are faster than my code, I am happy to do further optimization.
3. robust. many numerical stability techniques are applied to avoid numerical underflow and overflow which often happens when dealing with high dimensional data
4. easy to learn. the code are heavily commented, and the reference formulas in PRML book are indicated for corresponding code lines
5. practical. the package is designed not only for users to learn the algorithms in the book, but also to facility ML reseearch. Many functions in this package are already among the top downloads in Matlab file exchange and widely used.

License
-------
Currently Released Under GPLv3


Contact
-------
sth4nth at gmail dot com


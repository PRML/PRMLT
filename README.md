Introduction
-------
This Matlab package implements machine learning algorithms described in the great textbook:
Pattern Recognition and Machine Learning by C. Bishop ([PRML](http://research.microsoft.com/en-us/um/people/cmbishop/prml/)).

It is written purely in Matlab language. It is self-contained. There is no external dependency.

Note: this package requires Matlab **R2016b** or latter, since it utilizes a new Matlab syntax called [Implicit expansion](https://cn.mathworks.com/help/matlab/release-notes.html?rntext=implicit+expansion&startrelease=R2016b&endrelease=R2016b&groupby=release&sortby=descending) (a.k.a. broadcasting). It also requires Statistics Toolbox (for some simple random number generator) and Image Processing Toolbox (for reading image data).

Design Goal
-------
* Succinct: The code is extremely compact. Minimizing code length is a major goal. As a result, the core of the algorithms can be easily spotted.
* Efficient: Many tricks to speedup Matlab code are applied (eg. vectorization, matrix factorization, etc.). Usually, functions in this package are orders faster than Matlab builtin ones (e.g. kmeans).
* Robust: Many tricks for numerical stability are applied, such as computing probability in log domain, square root matrix update to enforce matrix symmetry\PD, etc.
* Readable: The code is heavily commented. Corresponding formulas in PRML are annoted. Symbols are in sync with the book.
* Practical: The package is not only readable, but also meant to be easily used and modified to facilitate ML research. Many functions in this package are already widely used (see [Matlab file exchange](http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A49739)).

Installation
-------
1. Download the package to a local folder (e.g. ~/PRMLT/) by running: 
```console
git clone https://github.com/PRML/PRMLT.git
```
2. Run Matlab and navigate to the folder (~/PRMLT/), then run the init.m script.

3. Run some demos in ~/PRMLT/demo folder. Enjoy!

FeedBack
-------
If you find any bug or have any suggestion, please do file issues. I am graceful for any feedback and will do my best to improve this package.

License
-------
Released under MIT license

Contact
-------
sth4nth at gmail dot com

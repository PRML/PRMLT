% CHAPTER01
%   condEntropy      - Compute conditional entropy z=H(x|y) of two discrete variables x and y.
%   entropy          - Compute entropy z=H(x) of a discrete variable x.
%   jointEntropy     - Compute joint entropy z=H(x,y) of two discrete variables x and y.
%   mutInfo          - Compute mutual information I(x,y) of two discrete variables x and y.
%   nmi              - Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
%   nvi              - Compute normalized variation information z=(1-I(x,y)/H(x,y)) of two discrete variables x and y.
%   relatEntropy     - Compute relative entropy (a.k.a KL divergence) z=KL(p(x)||p(y)) of two discrete variables x and y.
% CHAPTER02    
%   logDirichlet     - Compute log pdf of a Dirichlet distribution.
%   logGauss         - Compute log pdf of a Gaussian distribution.
%   logKde           - Compute log pdf of kernel density estimator.
%   logMn            - Compute log pdf of a multinomial distribution.
%   logMvGamma       - Compute logarithm multivariate Gamma function 
%   logSt            - Compute log pdf of a Student's t distribution.
%   logVmf           - Compute log pdf of a von Mises-Fisher distribution.
%   logWishart       - Compute log pdf of a Wishart distribution.
% CHAPTER03    
%   linReg           - Fit linear regression model y=w'x+w0  
%   linRegFp         - Fit empirical Bayesian linear model with Mackay fixed point method (p.168)
%   linRegPred       - Compute linear regression model reponse y = w'*X+w0 and likelihood
%   linRnd           - Generate data from a linear model p(t|w,x)=G(w'x+w0,sigma), sigma=sqrt(1/beta) 
% CHAPTER04    
%   binPlot          - Plot binary classification result for 2d data
%   fda              - Fisher (linear) discriminant analysis
%   logitBin         - Logistic regression for binary classification optimized by Newton-Raphson method.
%   logitBinPred     - Prediction of binary logistic regression model
%   logitMn          - Multinomial regression for multiclass problem (Multinomial likelihood)
%   logitMnPred      - Prediction of multiclass (multinomial) logistic regression model
%   sigmoid          - Sigmod function
%   softmax          - Softmax function
% CHAPTER05
%   mlpClass         - Train a multilayer perceptron neural network for classification with backpropagation
%   mlpClassPred     - Multilayer perceptron classification prediction
%   mlpReg           - Train a multilayer perceptron neural network for regression with backpropagation
%   mlpRegPred       - Multilayer perceptron regression prediction
% CHAPTER06    
%   kn2sd            - Transform a kernel matrix (or inner product matrix) to a squared distance matrix
%   knCenter         - Centerize the data in the kernel space
%   knGauss          - Gaussian (RBF) kernel K = exp(-|x-y|/(2s));
%   knKmeans         - Perform kernel kmeans clustering.
%   knKmeansPred     - Prediction for kernel kmeans clusterng
%   knLin            - Linear kernel (inner product)
%   knPca            - Kernel PCA
%   knPcaPred        - Prediction for kernel PCA
%   knPoly           - Polynomial kernel k(x,y)=(x'y+c)^o
%   knReg            - Gaussian process (kernel) regression
%   knRegPred        - Prediction for Gaussian Process (kernel) regression model
%   sd2kn            - Transform a squared distance matrix to a kernel matrix. 
% CHAPTER07    
%   rvmBinFp         - Relevance Vector Machine (ARD sparse prior) for binary classification.
%   rvmBinPred       - Prodict the label for binary logistic regression model
%   rvmRegFp         - Relevance Vector Machine (ARD sparse prior) for regression
%   rvmRegPred       - Compute RVM regression model reponse y = w'*X+w0 and likelihood 
%   rvmRegSeq        - Sparse Bayesian Regression (RVM) using sequential algorithm
% CHAPTER08    
%  MRF    
%   mrfBethe         - Compute Bethe energy
%   mrfBp            - Undirected graph belief propagation for MRF
%   mrfGibbs         - Compute Gibbs energy
%   mrfIsGa          - Contruct a latent Ising MRF with Gaussian observation
%   mrfMf            - Mean field for MRF
%  NaiveBayes    
%   nbBern           - Naive bayes classifier with indepenet Bernoulli.
%   nbBernPred       - Prediction of naive Bayes classifier with independent Bernoulli.
%   nbGauss          - Naive bayes classifier with indepenet Gaussian
%   nbGaussPred      - Prediction of naive Bayes classifier with independent Gaussian.
% CHAPTER09    
%   kmeans           - Perform kmeans clustering.
%   kmeansPred       - Prediction for kmeans clusterng
%   kmeansRnd        - Generate samples from a Gaussian mixture distribution with common variances (kmeans model).
%   kmedoids         - Perform k-medoids clustering.
%   kseeds           - Perform kmeans++ seeding
%   linRegEm         - Fit empirical Bayesian linear regression model with EM (p.448 chapter 9.3.4)
%   mixBernEm        - Perform EM algorithm for fitting the Bernoulli mixture model.
%   mixBernRnd       - Generate samples from a Bernoulli mixture distribution.
%   mixGaussEm       - Perform EM algorithm for fitting the Gaussian mixture model.
%   mixGaussPred     - Predict label and responsibility for Gaussian mixture model.
%   mixGaussRnd      - Genarate samples form a Gaussian mixture model.
%   rvmBinEm         - Relevance Vector Machine (ARD sparse prior) for binary classification.
%   rvmRegEm         - Relevance Vector Machine (ARD sparse prior) for regression
% CHAPTER10
%   linRegVb         - Variational Bayesian inference for linear regression.
%   mixGaussEvidence - Variational lower bound of the model evidence (log of marginal likelihood)
%   mixGaussVb       - Variational Bayesian inference for Gaussian mixture.
%   mixGaussVbPred   - Predict label and responsibility for Gaussian mixture model trained by VB.
%   rvmRegVb         - Variational Bayesian inference for RVM regression.
% CHAPTER11
%   dirichletRnd     - Generate samples from a Dirichlet distribution.
%   discreteRnd      - Generate samples from a discrete distribution (multinomial).
%   Gauss            - Class for Gaussian distribution used by Dirichlet process
%   gaussRnd         - Generate samples from a Gaussian distribution.
%   GaussWishart     - Class for Gaussian-Wishart distribution used by Dirichlet process
%   mixDpGb          - Collapsed Gibbs sampling for Dirichlet process (infinite) mixture model. 
%   mixDpGbOl        - Online collapsed Gibbs sampling for Dirichlet process (infinite) mixture model. 
%   mixGaussGb       - Collapsed Gibbs sampling for Dirichlet process (infinite) Gaussian mixture model (a.k.a. DPGM). 
%   mixGaussSample   - Genarate samples form a Gaussian mixture model with GaussianWishart prior.
% CHAPTER12 
%   fa               - Perform EM algorithm for factor analysis model
%   pca              - Principal component analysis
%   pcaEm            - Perform EM-like algorithm for PCA (by Sam Roweis).
%   pcaEmC           - Perform Constrained EM like algorithm for PCA.
%   ppcaEm           - Perform EM algorithm to maiximize likelihood of probabilistic PCA model.
%   ppcaRnd          - Generate data from probabilistic PCA model
%   ppcaVb           - Perform variatioanl Bayeisan inference for probabilistic PCA model. 
% CHAPTER13 
%  HMM 
%   hmmEm            - EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
%   hmmFilter        - HMM forward filtering algorithm. 
%   hmmRnd           - Generate a data sequence from a hidden Markov model.
%   hmmSmoother      - HMM smoothing alogrithm (normalized forward-backward or normalized alpha-beta algorithm).
%   hmmViterbi       - Viterbi algorithm (calculated in log scale to improve numerical stability).
%  LDS 
%   kalmanFilter     - Kalman filter (forward algorithm for linear dynamic system)
%   kalmanSmoother   - Kalman smoother (forward-backward algorithm for linear dynamic system)
%   ldsEm            - EM algorithm for parameter estimation of linear dynamic system.
%   ldsPca           - Subspace method for learning linear dynamic system.
%   ldsRnd           - Generate a data sequence from linear dynamic system.
% CHAPTER14 
%   adaboostBin      - Adaboost for binary classification (weak learner: kmeans)
%   adaboostBinPred  - Prediction of binary Adaboost
%   mixLinPred       - Prediction function for mxiture of linear regression
%   mixLinReg        - Mixture of linear regression
%   mixLinRnd        - Generate data from mixture of linear model
%   mixLogitBin      - Mixture of logistic regression model for binary classification optimized by Newton-Raphson method
%   mixLogitBinPred  - Prediction function for mixture of logistic regression

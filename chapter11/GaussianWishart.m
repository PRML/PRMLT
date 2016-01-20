
classdef GaussianWishart
     properties
         nu_
         kappa_
         m_
         W_
     end
     
     methods
         function obj = GaussianWishart(prior)
             obj.kappa_ = prior.kappa;
             obj.m_ = prior.m;
             obj.nu_ = prior.nu;
             obj.W_ = prior.W;
         end
        
         function obj = addSample(obj, X)
             kappa0 = obj.kappa_;
             m0 = obj.m_;
             nu0 = obj.nu_;
             W0 = obj.W_;
             
             n = size(X,2);
             xbar = mean(X,2);
             kappa = kappa0+n;
             m = (kappa0*m0+n*xbar)/kappa;
             xm = xbar-m0;
             W = W0+X*X'+xm*xm'*kappa*n/(kappa0+n);
             nu = nu0+n;             

             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.W_ = W;
         end
         
          function obj = delSample(obj, X)
             kappa0 = obj.kappa_;
             m0 = obj.m_;
             nu0 = obj.nu_;
             W0 = obj.W_;
             
             n = size(X,2);
             xbar = mean(X,2);
             kappa = kappa0+n;
             m = (kappa0*m0+n*xbar)/kappa;
             xm = xbar-m0;
             W = W0+X*X'+xm*xm'*kappa*n/(kappa0+n);
             nu = nu0+n;             

             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.W_ = W;
          end
         
        function p = predict(obj, X)
             kappa0 = obj.kappa_;
             m0 = obj.m_;
             nu0 = obj.nu_;
             W0 = obj.W_;
             
       
         end
     end
end
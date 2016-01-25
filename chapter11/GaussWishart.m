classdef GaussWishart
        properties
         kappa_
         mu_
         nu_
         U_
     end
     
     methods
         function obj = GaussWishart(kappa,mu,nu,W)
             U = chol(W);
             
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
        
         function obj = addSample(obj, x)
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;
             
             kappa = kappa+1;
             mu = mu+(x-mu)/n;
             nu = nu+1;
             U = cholupdate(U,x,'+');
             
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = delSample(obj, x)
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;

             kappa = kappa-1;
             mu = mu-(x-mu)/n;
             nu = nu-1;
             U = cholupdate(U,x,'-');
             
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function y = logPredPdf(obj,X)
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;
                
         end
     end
end

classdef GaussWishart
        properties
         kappa_
         m_
         nu_
         U_
     end
     
     methods
         function obj = GaussWishart(kappa,m,nu,S)
             U = chol(S+kappa*m*m');
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
        
         function obj = addSample(obj, x)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;
             
             kappa = kappa+1;
             m = m+(x-m)/kappa;
             nu = nu+1;
             U = cholupdate(U,x,'+');
             
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = delSample(obj, x)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;

             kappa = kappa-1;
             m = m-(x-m)/kappa;
             nu = nu-1;
             U = cholupdate(U,x,'-');
             
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function y = logPredPdf(obj,X)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;
             
             d = size(X,1);
             v = (nu-d+1);
             r = (1+1/kappa)/v;
             U = cholupdate(U,sqrt(kappa)*m,'-')*sqrt(r);
             
             X = bsxfun(@minus,X,m);
             Q = U'\X;
             q = dot(Q,Q,1);  % quadratic term (M distance)
             o = -log(1+q/v)*((v+d)/2);
             c = gammaln((v+d)/2)-gammaln(v/2)-(d*log(v*pi)+2*sum(log(diag(U))))/2;
             y = c+o;
         end
     end
end

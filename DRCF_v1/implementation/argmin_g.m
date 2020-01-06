function T = argmin_g(w0, rho, lambda, mu, X, T)
     lhd= 1 ./  ((lambda + rho) * w0 + mu); % left hand
     
     % compute T for each channel
     for i = 1:size(X,3)
         T(:,:,i) = lhd .* X(:,:,i);
     end
end



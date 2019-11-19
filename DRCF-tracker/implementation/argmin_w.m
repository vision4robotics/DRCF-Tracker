function T = argmin_w (w0, rho, lambda , X, T)
     lhd= 1 ./  (w0 .^2 + rho + lambda ); % left hand
     
     % compute T for each channel
     for i = 1:size(X,3)
         T(:,:,i) = lhd .* X(:,:,i);
     end
end

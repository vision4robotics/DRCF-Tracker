function T = argmin_w (s1, s2, rho , X, T)
     lhd= 1 ./  ...
         ((s2 .*s2) + (rho * s1.*s1)); 
     
     % compute T for each channel
     for i = 1:size(X,3)
         T(:,:,i) = lhd .* X(:,:,i);
     end
end

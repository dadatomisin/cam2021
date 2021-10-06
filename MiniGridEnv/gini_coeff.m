% x = mean(reward)./[25,22,15,15,18];
% val = 0;
% for i = 1:5
%     for j = 1:5
%         val = val + abs(x(i) - x(j));
%     end
% end
% val/(2*5*sum(x))

sum(mean(reward))/95
(mean(reward)./[25,22,15,15,18])
mean(mean(reward)./[25,22,15,15,18])
gini(mean(reward)./[25,22,15,15,18])
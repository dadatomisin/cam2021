n = 5;
battery = battery';
action = action';
demand = demand';
reward = reward';
t1 = [1, 3, 5, 7, 9];
t2 = [2, 4, 6, 8, 10];
scale = [25, 22, 15, 15, 18];

figure(1)
for i = 1:n
    subplot(n, 2, t1(i))
    %subplot(2, n, t1(i))
    t = 0:1:9999;
    plot(t, battery(i,:),'g')
    hold on
    plot(t, action(i,:), '-*')
    hold on
    plot(t, demand(i,:), 'r')
    hold on
    %plot(t, target(i,1:1000))
    legend('battery','action','demand')%, 'target')
    %legend('battery','demand')
    hold off
    grid on 
    grid minor
    ymax = max(100, max(demand(i,1:100))*1.2);
    axis([0, 100, 0, 100])
    
    subplot(n, 2, t2(i))
    %subplot(2, n, t2(i))
    t = 0:1:9999;
    R = reward(i,:)/scale(i);
    mR = mean(R)*ones(1,10000);
    plot(t, R, 'b')
    hold on
    plot(t, mR, 'k')
    legend('reward', 'average reward')
    hold off
    grid on 
    grid minor
    ymin = min(R(1:100)) - 0.2;
    ymax = max(R(1:100)) + 0.2;
    axis([0, 100, ymin, ymax])
end

figure(2)
n = 1;
subplot(5,1,1)
histogram(action(1,:),'Normalization','probability','BinWidth',n)
subplot(5,1,2)
histogram(action(2,:),'Normalization','probability','BinWidth',n)
subplot(5,1,3)
histogram(action(3,:),'Normalization','probability','BinWidth',n)
subplot(5,1,4)
histogram(action(4,:),'Normalization','probability','BinWidth',n)
subplot(5,1,5)
histogram(action(5,:),'Normalization','probability','BinWidth',n)

x = 0.0001:0.0001:0.9999;
y = log(x/(1-x));

% battery = battery';
% action = action';
% demand = demand';
% reward = reward';
% t1 = [1, 3, 5, 7, 9];
% t2 = [2, 4, 6, 8, 10];
% scale = [25, 22, 15, 15, 18];
% 
% figure('DefaultAxesFontSize',18)
% i = 2;
% t = 0:1:9999;
% plot(t, battery(2,:),'g')
% hold on
% plot(t, action(2,:), 'b-*')
% hold on
% plot(t, demand(2,:), 'r')
% hold on
% %plot(t, target(i,1:1000))
% legend('battery','action','demand')%, 'target')
% %legend('battery','demand')
% hold off
% % grid on 
% % grid minor
% ymax = max(100, max(demand(2,1:100))*1.2);
% axis([0, 100, 0, 100])
% title(['Agent ', num2str(i)])
% xlabel('Timestep')
function val = metrics(avg, mu)
    %tot_mu = sum(mu')';
    val = zeros(1,6);
    wac = sum(avg, 2)./sum(mu, 2);
    wai = (avg./mu);
    tot_wai = sum(wai,2)/5; % changed to mean by adding /5
    eq = zeros(10,1);
    for i = 1:10
        eq(i) = gini(wai(i,:));
    end
    val(1) = mean(wac); val(3) = mean(tot_wai); val(5) = mean(eq);
    val(2) = std(wac); val(4) = std(tot_wai); val(6) = std(eq);
end
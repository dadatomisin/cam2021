function y = gini(x)
    y = 0;
    for i = 1:5
        for j = 1:5
            y = y + abs(x(i) - x(j));
        end
    end
    y = y/(2*5*sum(x));
end
import numpy

def CR_converge(p, c, k, gamma=1, h=0, mu, n, battery, Q):
    z = []
    a = []
    new_a = []
    CR = []
    effective_c = []
    proportional_multiplier = []
    for i in range(n):
        proportional_multiplier.append(0) 
        effective_c.append(0)
        tmp = (p - k)/(p - (k + h))
        CR.append(tmp)
        z = cdf_inv(tmp)
        z_.append(cdf_inv(tmp))
        a.append(z - battery[i])
        new_a.append(max(z - battery[i], 0))

    iterations = 0
    converged = False
    while not converged:
        check = 0
        order_total = sum(a)
        excess = max(0, Q - order_total)
        for i in range(n):
            proportional_multiplier[i] = (excess/order_total)
            effective_c[i] = c * proportional_multiplier[i]
            CR[i] = (p - (gamma*effective_c[i] + k))/(p - (gamma*effective_c[i] + k + h))
            z[i] = cdf_inv(CR[i])
            new_a[i] = max(0, z[i])
            if new_a[i] - a[i] < 0.01:
                check += 1
        iterations += 1
        if check == n or iterations >= iterations_limit:
            converged = True
    
    return new_a[i]
            

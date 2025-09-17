power2 = [None]*10000
power2[0] = 1

def power2n(n):
    if not power2[n] is None: return power2[n]
    power2[n] = power2n(n-1)+power2n(n-1) 
    return power2[n]

n = 50
print(f'power2n({n}) = {power2n(n)}')
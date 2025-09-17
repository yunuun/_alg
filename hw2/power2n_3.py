def power2n(n):
    if n == 0: return 1
    return 2*power2n(n-1)

n = 50
print(f'power2n({n}) = {power2n(n)}')
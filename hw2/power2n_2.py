def power2n(n):
    if n == 0: return 1
    return power2n(n-1)+power2n(n-1)

n = 10
print(f'power2n({n}) = {power2n(n)}')
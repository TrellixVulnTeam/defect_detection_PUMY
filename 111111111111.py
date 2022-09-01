def test(j):
    return lambda i: i*j*j
val = test(3)
print(val(7))
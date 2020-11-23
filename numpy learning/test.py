def square(x):
    return x**2

result = []

#result = [square(x) for x in range(10)]

for i in range(10):
    if i%2 ==0:
        result.append(square(i))
    else: continue


print(result)
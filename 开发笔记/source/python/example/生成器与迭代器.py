#%%
# 生成器的注意事项：只能迭代一次
# gen = (i ** 2 for i in range(100))
# print(sum(gen))  # return 328350
# print(sum(gen))  # return 0

# PS：是迭代器的特性
ite = iter([i ** 2 for i in range(100)])
print(sum(ite))  # return 328350
print(sum(ite))  # return 0


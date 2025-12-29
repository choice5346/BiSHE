import jieba

words = jieba.lcut("我爱自然语言处理")
print(words)

jieba.add_word("自然语言处理")
words = jieba.lcut("我爱自然语言处理")  
print(words)

jieba.del_word("自然语言处理")
words = jieba.lcut("我爱自然语言处理")
print(words)
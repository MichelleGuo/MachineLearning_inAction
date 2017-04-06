import sys

# def main():
#     for line in sys.stdin:
#         a = line.split(",")
#         numa = int(a[0])
#         numb = int(a[1])
#         stra = a[0]
#         strb = a[1]
#         if(numa>70000 or numb>70000 or numa<1 or numb<1):
#             result=-1
#         else:
#             result=int(stra[::-1]) + int(strb[::-1])
#         return result

# if __name__ == "__main__":
#     a,b = raw_input().split(",")
#     result = 0
#     numa = int(a)
#     numb = int(b)
#     if(numa>70000 or numb>70000 or numa<1 or numb<1):
#         result=-1
#     else:
#         result=int(a[::-1]) + int(b[::-1])
# 	print result

if __name__ == "__main__":
    opt = raw_input().split()
    ini = "123456"
    # 0L 1R 2F 3B 4A 5C
    for i in range(len(opt)):
        step = opt[i]
        if step=='R':
            ini[0]='5'
            ini[1]='6'





import numpy as np

'''
# AND 게이트
def AND(x1, x2):
    w1 = 0.7
    w2 = 0.7
    theta = 1
    y = x1*w1 + x2*w2
    if y > theta:
        return 1
    else:
        return 0

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
'''
'''
# OR 게이트
def OR(x1, x2):
    w1 = 0.7
    w2 = 0.7
    theta = 0.5
    y = x1*w1 + x2*w2
    if y > theta:
        return 1
    else:
        return 0
'''

'''
# NAND 게이트
def NAND(x1, x2):
    w1 = 0.7
    w2 = 0.7
    theta = 1
    y = x1*w1 + x2*w2
    if y > theta:
        return 0
    else:
        return 1
'''

# 편향 도입(bias)
# numpy 사용

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.7, 0.7])
    b = - 1.0

    y = np.dot(x, w) + b
    if y > 0:
        return 1
    else:
        return 0

print(AND(1, 0))

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.7, 0.7])
    b = -0.1

    y = np.dot(x, w) + b

    if y > 0:
        return 1
    else:
        return 0

print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.7, -0.7])
    b = 1.0
    y = np.dot(x, w) + b

    if y > 0:
        return 1
    else:
        return 0

print(NAND(1, 1))

# XOR 게이트
# x1, x2 중 하나만 1일때 1을 출력

def XOR(x1, x2):
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)     
    y = AND(s1, s2)
    return y

print('dddd')
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
        
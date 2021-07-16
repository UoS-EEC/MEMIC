#!/usr/bin/env python3
import math
import numpy as np
import sys

def factor3(num):
    factors = []
    for i in range(1,num+1):
        if num % i != 0:
            continue
        for j in range(1,num+1):
            if (num//i) % j != 0:
                continue
            for k in range(1,num+1):
                if i*j*k == num:
                    factors.append((i, j, k))
    return factors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: Factor3 <number>")
        exit(1)
    print(factor3(int(sys.argv[1])))
    exit(0)

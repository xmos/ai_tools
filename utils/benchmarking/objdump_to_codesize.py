#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys

symbols = []

for line in sys.stdin:
    if line[15:20] == '.text':
        size = int(line[26:36], 0)
        if size > 0:
            symbol = line[37:].strip()
            symbols.append({
                'name': symbol, 
                'size': size
            })

symbols = sorted(symbols, key=lambda k: k['size']) 

total = 0

for symbol in symbols:
    name = symbol['name']
    size = symbol['size']
    total += size
    print(f'{size}     {name}')

#print(total)
''' 
Description:    Creating an AI that will predict what type the pokemon is based on the given stats.
Authors:        Isaac Garay and Riley Peters
Links:          https://www.kaggle.com/datasets/alopez247/pokemon
                https://bulbapedia.bulbagarden.net/wiki/Stat
'''
import math
import numpy as np

def main():
    pass

'''Desciption: Calculates the HP stat for the pokemon
Inputs: base stat, iv, ev, level, gen
Output: HP stat as an int'''
def hp(base, iv, ev, level, gen):
    if gen >= 3:
        return math.floor(((2 * base + iv + math.floor(ev/4)) * level)/100) + level + 10
    else:
        return math.floor((((base + iv) * 2 + math.floor(math.ceil(math.sqrt(ev))/4)) * level)/100) + level + 10

'''Desciption: Calculates any stat for the pokemon other than HP
Inputs: base stat, iv, ev, level, gen, nature (2: benifits stat, 1: netural, 0: hinders stat)
Output: Stat as an int'''
def stat(base, iv, ev, level, gen, nature):
    if gen >= 3:
        if nature == 2: nature = 1.1
        elif nature == 1: nature = 1
        else: nature = 0.9
        
        return math.floor((math.floor(((2 * base + iv + math.floor(ev/4)) * level) / 100) + 5) * nature)
    else:
        return math.floor((((base + iv) * 2 + math.floor(math.ceil(math.sqrt(ev))/4)) * level)/100) + 5

if __name__ == "__main__":
    main()
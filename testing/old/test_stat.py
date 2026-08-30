from argparse import ArgumentParser
from scipy.stats import wilcoxon


def main(): 
    _, pval = wilcoxon([0.828, 0.833, 0.755, 0.897, 0.882], 
                       [0.752, 0.646, 0.801, 0.725, 0.676])

    print(pval)

if __name__ == '__main__': 
    main()
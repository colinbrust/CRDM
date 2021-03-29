from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
parser = importr('fstparsr')

dates = robjects.StrVector(['20120101, 20130103'])
f = '/Users/colinbrust/Box/school/Data/drought/in_features/fst/pr.fst'
parser = importr('fstparsr')
row_idx = [x for x in range(1, 17)]
row_idx = robjects.IntVector(row_idx)
col_idx = 1

a = parser.index_fst(f, dates, row_idx, col_idx)
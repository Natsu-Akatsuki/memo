# Match&&Assignment

### linear sum assignment problem

 A problem instance is described by a matrix C, where each C[i,j] is the cost of matching vertex i of the first partite set (a “worker”) and vertex j of the second set (a “job”). The goal is to find a complete assignment of workers to jobs of minimal cost.

```python
import numpy as np
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)
matched_indices = np.stack((row_ind, col_ind), axis=1)
# col_ind: array([1, 0, 2]), row_ind: array([0, 1, 2])
# cost[row_ind, col_ind].sum(): 5
```


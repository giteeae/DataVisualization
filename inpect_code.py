from sklearn.model_selection import train_test_split
from rich import print

import inspect
lines = inspect.getsource(train_test_split)
print(lines)

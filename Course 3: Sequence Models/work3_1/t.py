import trax
from trax.fastmath import grad

n_to_take = 2
target_pos = [1] * n_to_take
target_neg = [0] * n_to_take
target_l = target_pos + target_neg
print(target_l)
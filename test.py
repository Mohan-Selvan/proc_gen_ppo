# import numpy as np
# import math

# from_range = (0, 2)
# to_range = (0, 255)

# def remap(number, from_range, to_range):
#     input_start, input_end = from_range
#     output_start, output_end = to_range
#     output = output_start + ((output_end - output_start) / (input_end - input_start)) * (number - input_start)
#     return output

# grid = np.ones((3, 5), dtype=np.uint8)

# grid[(2, 2)] = 0
# grid[(1, 3)] = 2
# print(np.round((grid / 2) * 255).astype(np.uint8))

# from gymnasium.envs.registration import register

# register(
#     id='GameWorld-v0',  # Unique ID for your environment
#     entry_point='GameWorld',  # Full path to your class
# )


# from gymnasium import envs
# all_envs = envs.registry.keys()
# print(sorted(list(all_envs)))

import numpy as np

arr = np.random.uniform(low=-1.0, high=1.0, size=(1,25))

def remap_array(array, from_range, to_range):
    for index, num in enumerate(array):
        array[index] = remap(num, from_range, to_range)
    
    return array

def remap(value, from_range, to_range):
    from_min, from_max = from_range
    to_min, to_max = to_range
    remapped_value = (((value - from_min) * (to_max - to_min)) / ((from_max - from_min))) + to_min
    return remapped_value

print(arr)

arr = remap_array(arr, (-1.0, 1.0), (0, 1))
print(arr)
arr = np.round(arr).astype(int)
print(arr)
arr = arr.reshape((5, 5))
print(arr)
print('------------------')
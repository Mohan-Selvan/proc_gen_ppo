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

from gymnasium.envs.registration import register

register(
    id='GameWorld-v0',  # Unique ID for your environment
    entry_point='GameWorld',  # Full path to your class
)


from gymnasium import envs
all_envs = envs.registry.keys()
print(sorted(list(all_envs)))

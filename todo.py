# Change data type of Obs space to float
# Use Add a different channel containing tile counts.
# Add something with respect to the player simulation in the observation space.
# Use diagonal instead of a 2D mask.
# Use single step learning using a big observation space and action mask
# Invert and try
# Consider starting and end point within the mask
# Create random point in the mask and ensure the cell is reachable.


# Last used action space in observation matrix

# Check determinism in prediction? Test with deterministic set to False
# Save paths and according playable levels during training.

# Try one hot encoding of 5x5 mask and DQN
# Try TRPO

# Find last actionable cell in player path dynamically
# Test with PPO, RecPPO and TRPO with 2 actions, 5 step forward
# Move agent forward even irrespective of reachability condition when testing.
# Implement increasingly complex path.
# Check hidden layer parameter in policy dict.

# Add a different tile for default tile
# Remap input values from -1 to 1
# Adjust max_step_count

# Check evaluate_policy from sb3

# Best model callback
# Increase padding
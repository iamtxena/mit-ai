import numpy as np

# Create a 1D array
array_1d = np.array([2, 5, 1, 8, 3])
print("Original Array:", array_1d)

# Sort the array
sorted_array = np.sort(array_1d)
print("Sorted Array:", sorted_array)

# Create a 2D array (matrix)
array_2d = np.array([[3, 8, 1], [2, 0, 5]])
print("Original Matrix:\n", array_2d)

# Find the minimum and maximum values in the matrix
min_value = np.min(array_2d)
max_value = np.max(array_2d)
print("Minimum value:", min_value)
print("Maximum value:", max_value)

# Calculate the sum of a specific column (e.g., the second column)
column_sum = np.sum(array_2d[:, 1])
print("Sum of the second column:", column_sum)

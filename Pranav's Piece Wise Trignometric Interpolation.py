import numpy as np
import matplotlib.pyplot as plt

# Function to generate trigonometric interpolation for a single bounce
def generate_interpolation(x_points, y_points):
    n = len(x_points)
    A = np.array([["1", f"cos({x:.3f})", f"sin({x:.3f})"] for x in x_points], dtype=object)

    # Print equations in regular and Unicode formats
    print("Equations to Solve (Regular and Unicode Format):")
    for i, (row, y) in enumerate(zip(A, y_points)):
        regular_equation = f"{y:.3f} = a0 + {row[1]}*a1 + {row[2]}*b1"
        unicode_equation = f"{y:.3f} = a₀ + {row[1]}·a₁ + {row[2]}·b₁"
        print(f"Regular: {regular_equation}")
        print(f"Unicode: {unicode_equation}")

    # Print the matrix system in regular and Unicode forms
    print("\nMatrix Representation:")
    print("Matrix A (Regular):")
    print(A)
    print("Matrix y (Regular):")
    print(y_points)

    print("\nMatrix A (Unicode):")
    matrix_a_unicode = "\n".join([
        "[ " + "  ".join(row) + " ]" for row in A
    ])
    print(matrix_a_unicode)

    print("\nMatrix y (Unicode):")
    matrix_y_unicode = "[ " + "  ".join([f"{elem:.3f}" for elem in y_points]) + " ]"
    print(matrix_y_unicode)

    coefficients = np.linalg.solve(
        np.array([[1, np.cos(x), np.sin(x)] for x in x_points]), y_points
    )
    return coefficients

# Function to evaluate the trigonometric interpolation
def evaluate_interpolation(x, coefficients):
    a0, a1, b1 = coefficients
    return a0 + a1 * np.cos(x) + b1 * np.sin(x)

# Define the number of bounces
num_bounces = 5  # Change this variable to add more bounces

# Define the time and height points for each bounce (adjust as needed)
bounce_data = [
    (np.array([-0.468, 0, 0.468]), np.array([0, 1.073, 0])),  # First bounce
    (np.array([0.468, 0.813, 1.158]), np.array([0, 0.584, 0])),   # Second bounce
    (np.array([1.158, 1.421, 1.684]), np.array([0, 0.339, 0])),    # Third bounce
    (np.array([1.684, 1.884, 2.084]), np.array([0, 0.196, 0])),
    (np.array([2.084, 2.235, 2.386]), np.array([0, 0.112, 0]))
]

# Truncate bounce_data to the number of desired bounces
bounce_data = bounce_data[:num_bounces]

# Generate and plot each bounce
x_values = np.linspace(0, 3, 1000)  # Define a smooth range of x values
plt.figure(figsize=(10, 6))

for i, (x_points, y_points) in enumerate(bounce_data):
    coefficients = generate_interpolation(x_points, y_points)
    y_values = evaluate_interpolation(x_values, coefficients)

    # Apply limits: Zero out values outside the range of x_points
    x_min, x_max = x_points[0], x_points[-1]
    y_values_limited = np.where((x_values >= x_min) & (x_values <= x_max), y_values, 0)

    # Plot the interpolation function within its limits
    plt.plot(x_values, y_values_limited, label=f"Bounce {i + 1} Interpolation")

    # Scatter plot the original data points
    plt.scatter(x_points, y_points, label=f"Bounce {i + 1} Data Points")

    # Print the function for the current bounce and its range
    a0, a1, b1 = coefficients
    regular_function = f"f(x) = {a0:.3f} + {a1:.3f}*cos(x) + {b1:.3f}*sin(x)"
    unicode_function = f"f(x) = {a0:.3f} + {a1:.3f}·cos(x) + {b1:.3f}·sin(x)"
    print(f"Bounce {i + 1} Regular Function: {regular_function}")
    print(f"Bounce {i + 1} Unicode Function: {unicode_function}")
    print(f"Bounce {i + 1} Range: [{x_min:.3f}, {x_max:.3f}]")

# Add plot details
plt.title("Trigonometric Interpolation for Multiple Bounces with Limits")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.legend()
plt.grid()
plt.show()

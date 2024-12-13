import numpy as np
import matplotlib.pyplot as plt

# Function to generate trigonometric interpolation for a single bounce
def generate_interpolation(x_points, y_points):
    n = len(x_points)
    if n != 11:
        raise ValueError("The number of x_points must match the number of coefficients (11).")
    A = np.array([
        [1, np.cos(1 * x), np.sin(1 * x), np.cos(2 * x), np.sin(2 * x), np.cos(3 * x), np.sin(3 * x), np.cos(4 * x), np.sin(4 * x), np.cos(5 * x), np.sin(5 * x)] for x in x_points
    ])
    print("Equations to Solve (in LaTeX):")
    for i, (row, y) in enumerate(zip(A, y_points)):
        equation = f"a_0"
        for j in range(1, len(row), 2):
            equation += f" + a_{{{(j+1)//2}}}\cos({(j+1)//2}x_{{{i+1}}}) + b_{{{(j+1)//2}}}\sin({(j+1)//2}x_{{{i+1}}})"
        print(f"Equation {i+1}: {equation} = {y:.3f}")
    coefficients = np.linalg.solve(A, y_points)
    return coefficients

# Function to evaluate the trigonometric interpolation
def evaluate_interpolation(x, coefficients):
    a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = coefficients
    return (a0 + a1 * np.cos(1 * x) + b1 * np.sin(1 * x) +
                a2 * np.cos(2 * x) + b2 * np.sin(2 * x) +
                a3 * np.cos(3 * x) + b3 * np.sin(3 * x) +
                a4 * np.cos(4 * x) + b4 * np.sin(4 * x) +
                a5 * np.cos(5 * x) + b5 * np.sin(5 * x))

# Define the number of bounces 
num_bounces = 1  # Adjust as needed

# Define the x-values (time points) and height points for each bounce
x_values_per_bounce = np.array([
    -0.468, 0, 0.468, 0.813, 1.158, 1.421, 1.684, 1.884, 2.084, 2.084 + 0.302 / 2, 2.386
    ])
y_values_per_bounce = np.array([
    0, 1.073, 0, -0.584, 0, 0.339, 0, -0.196, 0, 0.112, 0
    ])

bounce_data = [(x_values_per_bounce, y_values_per_bounce)]

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
    plt.plot(x_values, y_values_limited, label=f"Bouncing continious Interpolation")

    # Scatter plot the original data points
    plt.scatter(x_points, y_points, label=f"Bouncing Data Points")

    # Print the function for the current bounce and its range
    a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = coefficients
    print(f"Bounce {i + 1} Function (in LaTeX): f(x) = {a0:.3f} + {a1:.3f} \cos(1x) + {b1:.3f} \sin(1x) + {a2:.3f} \cos(2x) + {b2:.3f} \sin(2x) + {a3:.3f} \cos(3x) + {b3:.3f} \sin(3x) + {a4:.3f} \cos(4x) + {b4:.3f} \sin(4x) + {a5:.3f} \cos(5x) + {b5:.3f} \sin(5x)")
    print(f"Bounce {i + 1} Range: [{x_min:.3f}, {x_max:.3f}]")

print("Desmos-Compatible Function:")
terms = [f"{coefficients[0]:.16f}"]
terms += [f"{coefficients[i]:.16f}*cos({(i+1)//2}*x)" if i % 2 == 1 else f"{coefficients[i]:.16f}*sin({(i+1)//2}*x)" for i in range(1, len(coefficients))]
function_string = " + ".join(terms)
print(f"f(x) = {function_string}")

# Add plot details
plt.title("Trigonometric Interpolation for Multiple Bounces with Limits")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.legend()
plt.grid()
plt.show()

# Formula used for interpolation (in LaTeX):
print("Interpolation Formula (in LaTeX):")
print(r"f(x) = a_0 + \sum_{k=1}^{n} \left( a_k \cos(kx) + b_k \sin(kx) \right)")

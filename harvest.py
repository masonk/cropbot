import numpy as np
import matplotlib.pyplot as plt

# Define the transition matrix P (right-stochastic)
P = np.array([
    [0.75, 0.25, 0.00, 0.00],
    [0.00, 0.80, 0.20, 0.00],
    [0.00, 0.00, 0.97, 0.03],
    [0.00, 0.00, 0.00, 1.00]
])

# Initial state vector
v0 = np.array([1, 0, 0, 0])

seeds_per_plot = 23
value_vector = np.array([0.0073 * 12.7, 1.85 * 12.7, 47 * 12.7, 234 * 12.7 / 2]) * seeds_per_plot

# Number of transitions to compute
max_n = 10

# Store expected values and state distributions
n_values = list(range(max_n + 1))
expected_values = []
state_distributions = []

# Compute expected value and state distribution for each n
current_state = v0.copy()
for n in range(max_n + 1):
    # Calculate expected value: state_vector · value_vector
    ev = np.dot(current_state, value_vector)
    expected_values.append(ev)
    
    # Store state distribution (probability of being in each tier)
    state_distributions.append(current_state.copy())
    
    # Update state for next iteration
    if n < max_n:
        current_state = current_state @ P

# Calculate deltas (change in expected value after each transition)
deltas = [0]  # No delta for n=0
for i in range(1, len(expected_values)):
    deltas.append(expected_values[i] - expected_values[i-1])

# Create subplot figure
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))

# First chart - Expected Values and Deltas
line1 = ax1.plot(n_values, expected_values, 'b-', linewidth=2, label='Expected Value', marker='o', markersize=6)

# Add value labels for expected values
for i, (n, ev) in enumerate(zip(n_values, expected_values)):
    ax1.annotate(f'{ev:.0f}', 
                xy=(n, ev), 
                xytext=(0, 15),  # 15 points above the point
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

# Create second y-axis for deltas
ax2 = ax1.twinx()
line2 = ax2.plot(n_values, deltas, 'r--', linewidth=2, label='Delta (Change in EV)', marker='s', markersize=5)

# Add delta labels
for i, (n, delta) in enumerate(zip(n_values, deltas)):
    if delta != 0:  # Don't label the first point (delta = 0)
        ax2.annotate(f'{delta:.0f}', 
                    xy=(n, delta), 
                    xytext=(0, -15),  # 15 points below the point
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))

# Set labels and limits for first chart
ax1.set_xlabel('# Plot Upgrades (n)', fontsize=12)
ax1.set_ylabel('Expected Lifeforce Per Plot', fontsize=12, color='blue')
ax2.set_ylabel('Delta Next Upgrade', fontsize=12, color='red')
ax1.set_title('Crop Rotation Lifeforce', fontsize=14, fontweight='bold')

# Set y-axis limits
ax1.set_ylim(0, 15000)  # Set max to 15k
ax2.set_ylim(min(deltas) - 100, max(deltas) + 100)

# Grid and legend for first chart
ax1.grid(True, alpha=0.3)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Second chart - Expected Monsters Per Plot by Tier
colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']  # Orange, Green, Blue, Red
tier_labels = ['T1', 'T2', 'T3', 'T4']

# Calculate expected monsters per plot for each tier (state probability * seeds_per_plot)
for tier in range(4):
    expected_monsters = [state_distributions[n][tier] * seeds_per_plot for n in range(max_n + 1)]
    ax3.plot(n_values, expected_monsters, 
             color=colors[tier], 
             linewidth=2, 
             label=f'{tier_labels[tier]} Monsters', 
             marker='o', 
             markersize=5)

# Set labels and formatting for second chart
ax3.set_xlabel('# Plot Upgrades (n)', fontsize=12)
ax3.set_ylabel('Expected # Monsters Per Plot', fontsize=12)
ax3.set_title('Expected Monster Distribution by Tier', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='center right')

# Set reasonable y-axis limits for monster chart
ax3.set_ylim(0, seeds_per_plot + 2)

plt.tight_layout()

# Save the plot instead of showing it (for non-interactive environments)
plt.savefig('markov_expected_value_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'markov_expected_value_chart.png'")

# Try to show the plot (will work in interactive environments)
try:
    plt.show()
except:
    print("Note: Running in non-interactive mode - chart saved to file instead")

# Print some key values
print(f"Scaled value vector: [{value_vector[0]:.2f}, {value_vector[1]:.2f}, {value_vector[2]:.2f}, {value_vector[3]:.2f}]")
print("\nExpected Values and Deltas for n=0 to n=10:")
print("n    Expected Value    Delta")
print("-" * 35)
for n in range(11):
    print(f"{n:2d}   {expected_values[n]:12.2f}   {deltas[n]:8.2f}")

# Also show the state distributions at each step
print("\nState Distributions for n=0 to n=10:")
for n in range(11):
    if n == 0:
        state = v0
    else:
        state = v0 @ np.linalg.matrix_power(P, n)
    print(f"n={n:2d}: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}, {state[3]:.4f}]")

# Print expected monsters per tier
print(f"\nExpected Monsters Per Plot by Tier (seeds_per_plot = {seeds_per_plot}):")
print("n    T1 Monsters    T2 Monsters    T3 Monsters    T4 Monsters")
print("-" * 65)
for n in range(11):
    t1_monsters = state_distributions[n][0] * seeds_per_plot
    t2_monsters = state_distributions[n][1] * seeds_per_plot
    t3_monsters = state_distributions[n][2] * seeds_per_plot
    t4_monsters = state_distributions[n][3] * seeds_per_plot
    print(f"{n:2d}   {t1_monsters:10.2f}   {t2_monsters:10.2f}   {t3_monsters:10.2f}   {t4_monsters:10.2f}")
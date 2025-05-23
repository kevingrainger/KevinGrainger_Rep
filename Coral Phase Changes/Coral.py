import numpy as np
import random
import math
import csv
import matplotlib.pyplot as plt

L = 24  # Lattice size
q = 3  # States: 1 = Coral, 2 = Macroalgae, 3 = Turf
beta_steps = 50
total_steps = 10000
thermalization = 5000
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
grid = np.random.randint(1, q + 1, size=(L, L))

# Synthetic environmental data (spatial SST and nutrient maps)
sst_map = np.random.normal(loc=28.0, scale=1.5, size=(L, L))  # Degrees Celsius
nutrient_map = np.random.normal(loc=0.5, scale=0.2, size=(L, L))  # mg/L

# To store temporal snapshots
temporal_snapshots = []

color_map = {1: (0.0, 0.5, 1.0),  # Coral - Blue
             2: (0.0, 0.8, 0.0),  # Macroalgae - Green
             3: (0.5, 0.5, 0.5)}  # Turf - Gray

def periodic_bc(pos):
    return pos % L

def initialize_grid():
    global grid
    grid = np.random.randint(1, q + 1, size=(L, L))

def environmental_bias(x, y):
    sst = sst_map[x][y]
    nutrients = nutrient_map[x][y]

    bias = {
        1: 0.0,  # Coral baseline
        2: 0.0,  # Macroalgae
        3: 0.0   # Turf
    }

    if sst > 29.0:
        bias[1] += (sst - 29.0) * 1.0

    if nutrients > 0.6:
        bias[2] -= (nutrients - 0.6) * 1.5
        bias[1] += (nutrients - 0.6) * 0.5

    return bias

def delta_energy(x, y, new_spin):
    old_spin = grid[x][y]
    if old_spin == new_spin:
        return 0

    old_S = 0
    new_S = 0
    for i in range(4):
        nx = periodic_bc(x + dx[i])
        ny = periodic_bc(y + dy[i])
        neighbor_spin = grid[nx][ny]

        if neighbor_spin == old_spin:
            old_S += 1
        if neighbor_spin == new_spin:
            new_S += 1

    bias = environmental_bias(x, y)
    delta_bias = bias.get(new_spin, 0) - bias.get(old_spin, 0)
    return (old_S - new_S) + delta_bias

def metropolis(beta):
    for _ in range(L * L):
        x = random.randint(0, L - 1)
        y = random.randint(0, L - 1)
        old_spin = grid[x][y]
        new_spin = random.randint(1, q)

        if new_spin == old_spin:
            continue

        dE = delta_energy(x, y, new_spin)
        if dE <= 0 or random.random() < math.exp(-beta * dE):
            grid[x][y] = new_spin

def calculate_magnetization():
    counts = np.bincount(grid.flatten(), minlength=q + 1)[1:]
    max_count = np.max(counts)
    return (q * max_count / (L * L) - 1.0) / (q - 1.0)

def save_grid_snapshot(step):
    rgb_grid = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            rgb_grid[i, j] = color_map[grid[i, j]]
    plt.imshow(rgb_grid, interpolation='nearest')
    plt.title(f"Spatial Distribution (Step {step})")
    plt.axis('off')
    plt.savefig(f"snapshot_step_{step}.png")
    plt.close()
    temporal_snapshots.append(rgb_grid.copy())

def run_simulation():
    initialize_grid()
    magnetizations = []

    with open("coral_potts_results.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Beta", "Magnetization", "Mag^2", "Variance"])

        for i in range(beta_steps):
            beta = 0.5 + i * (1.5 - 0.5) / beta_steps
            total_mag = 0
            total_mag_sq = 0

            for step in range(total_steps):
                metropolis(beta)
                if step >= thermalization:
                    m = calculate_magnetization()
                    total_mag += m
                    total_mag_sq += m * m
                if step % 2000 == 0 and step > 0:
                    save_grid_snapshot(step)

            steps = total_steps - thermalization
            mag_avg = total_mag / steps
            mag_sq_avg = total_mag_sq / steps
            variance = mag_sq_avg - mag_avg ** 2
            writer.writerow([beta, mag_avg, mag_sq_avg, variance])
            magnetizations.append((beta, mag_avg))

    return magnetizations

def plot_results(magnetizations):
    betas, mags = zip(*magnetizations)
    plt.plot(betas, mags, marker='o')
    plt.xlabel('Beta')
    plt.ylabel('Average Magnetization')
    plt.title('Coral Reef State Transition (Potts Model)')
    plt.grid(True)
    plt.show()

def show_temporal_evolution():
    fig, axs = plt.subplots(1, len(temporal_snapshots), figsize=(15, 3))
    for idx, snapshot in enumerate(temporal_snapshots):
        axs[idx].imshow(snapshot, interpolation='nearest')
        axs[idx].axis('off')
        axs[idx].set_title(f"T{(idx+1)*2000}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    random.seed()
    results = run_simulation()
    plot_results(results)
    show_temporal_evolution()
    print("Simulation complete. Spatial maps and results saved.")

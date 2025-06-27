import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter

# parameters
GRID_SIZE = 100
TOTAL_STEPS = 100
PLOT_EVERY = 5
BETA = 0.15  # Interaction strength 1/kT

OCEAN = 0
ROCK = 1
HEALTHY_CORAL = 2
BLEACHED_CORAL = 3

class ReefSimulation:
    def __init__(self, bleaching_thresh=30.0, dhd_decay=0.9, excel_file="Coral temps.xlsx"):
        self.bleaching_thresh = bleaching_thresh
        self.dhd_decay = dhd_decay
        self.excel_file = excel_file
        
        #Load temperature data
        self.load_temperature_data(excel_file)
        self.current_day = 0
        self.base_temp = np.mean(self.temp_data)
        
        #reef and grids
        self.coral, self.substrate = self.initialize_reef()
        self.temp_grid = self.initialize_temperature()
        self.dhd_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        self.population_history = {
            'healthy': [],
            'bleached': [],
            'bare_rock': [],
            'ocean': [],
            'target_bleached': []
        }

    def load_temperature_data(self, filepath):
        df = pd.read_excel(filepath)
        
        temp_col = next(col for col in df.columns if 'temp' in col.lower())
        bleach_col = next(col for col in df.columns if 'smooth' in col.lower() or 'bleach' in col.lower())

        self.temp_data = df[temp_col].values
        self.bleaching_data = df[bleach_col].values

        if np.max(self.bleaching_data) > 1:
            self.bleaching_data = self.bleaching_data / 100.0

        print(f"Loaded {len(self.temp_data)} days of data")
        print(f"Temperature range: {np.min(self.temp_data):.1f} - {np.max(self.temp_data):.1f}°C")


    def initialize_reef(self):
        coral = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        substrate = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        #reef shape
        center = GRID_SIZE // 2
        radius = GRID_SIZE // 3
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                noise = 0.3 * random.random()
                if dist < radius * (0.8 + noise):
                    substrate[i,j] = ROCK
                    # Initial coral coverage
                    if random.random() < 0.7:
                        coral[i,j] = HEALTHY_CORAL
        
        # corals grow near other corals
        for iteration in range(3000):
            x, y = random.randint(1, GRID_SIZE-2), random.randint(1, GRID_SIZE-2)
            if substrate[x,y] != ROCK:
                continue
                
            #counting healthy neighbors
            healthy_neighbors = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                if coral[x+dx, y+dy] == HEALTHY_CORAL:
                    healthy_neighbors += 1
            
            # probability increases with healthy neighbors
            if coral[x,y] == 0 and healthy_neighbors >= 2:
                recruitment_prob = 0.03 * (healthy_neighbors / 4.0)
                if random.random() < recruitment_prob:
                    coral[x,y] = HEALTHY_CORAL
            
            #Random mortality
            elif coral[x,y] == HEALTHY_CORAL and random.random() < 0.005:
                coral[x,y] = 0
        
        return coral, substrate

    def initialize_temperature(self):
        temp_grid = np.full((GRID_SIZE, GRID_SIZE), self.temp_data[0])
        #small variations
        variation = np.random.normal(0, 0.1, (GRID_SIZE, GRID_SIZE))
        return gaussian_filter(temp_grid + variation, sigma=1)

    def update_temperature(self):
        day_temp = self.temp_data[self.current_day % len(self.temp_data)]
        
        #temperature field
        spatial_noise = np.random.normal(0, 0.2, (GRID_SIZE, GRID_SIZE))
        self.temp_grid = gaussian_filter(day_temp + spatial_noise, sigma=1.5)
        
        
        heat_excess = np.maximum(self.temp_grid - self.bleaching_thresh, 0)
        self.dhd_grid = self.dhd_grid * self.dhd_decay + heat_excess
        
        self.current_day += 1
        
        self.base_temp = day_temp

    def get_neighbors(self, x, y):
    
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
            neighbors.append(self.coral[nx, ny])
        return neighbors

    def calculate_potts_energy(self, x, y, state):
       #calculate Potts model energy for a given state at position
        if self.substrate[x,y] != ROCK:
            return 0
        
        energy = 0
        temp = self.temp_grid[x,y]
        neighbors = self.get_neighbors(x, y)
        
        # Potts model: energy decreases when neighbors have same state
        for neighbor_state in neighbors:
            if neighbor_state == state and neighbor_state != 0:  
                energy -= 1.0  # coupling
        
        
        if state == HEALTHY_CORAL:
            # penalty high temperature and accumulated heat stress
            if temp > self.bleaching_thresh:
                stress_factor = (temp - self.bleaching_thresh) * (1 + 0.05 * self.dhd_grid[x,y])
                energy += 2.0 * stress_factor
            else:
                energy -= 0.5  
                
        elif state == BLEACHED_CORAL:
            # penalty for being bleached
            energy += 0.5
            # Recovery with cool temperatures and healthy neighbors
            if temp < self.bleaching_thresh - 0.5:
                healthy_neighbors = neighbors.count(HEALTHY_CORAL)
                energy -= 0.3 * healthy_neighbors
        
        return energy

    def metropolis_step(self):
        #Metropolis algorithm for Potts model
        new_coral = self.coral.copy()
        
        
        for _ in range(GRID_SIZE * GRID_SIZE):
            
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            
            if self.substrate[x,y] != ROCK:
                continue
                
            current_state = self.coral[x,y]
            
            
            possible_states = self.get_allowed_transitions(current_state, x, y)
            if len(possible_states) <= 1:
                continue
                
            new_state = random.choice(possible_states)
            if new_state == current_state:
                continue
            
            #energy difference
            old_energy = self.calculate_potts_energy(x, y, current_state)
            new_energy = self.calculate_potts_energy(x, y, new_state)
            delta_E = new_energy - old_energy
            
            # acceptance criterion
            if delta_E <= 0:
                # accept move if lower energy
                new_coral[x,y] = new_state
            else:
                # accept with probability exp(-β * ΔE)
                acceptance_prob = np.exp(-BETA * delta_E)
                if random.random() < acceptance_prob:
                    new_coral[x,y] = new_state
        
        self.coral = new_coral

    def get_allowed_transitions(self, current_state, x, y):
        #allowed state transitions for a coral
        neighbors = self.get_neighbors(x, y)
        healthy_neighbors = neighbors.count(HEALTHY_CORAL)
        temp = self.temp_grid[x,y]
        
        if current_state == HEALTHY_CORAL:
            # healthy coral can stay healthy or bleach
            transitions = [HEALTHY_CORAL]
            # bleaching probability with temperature and DHD
            if temp > self.bleaching_thresh:
                transitions.append(BLEACHED_CORAL)
            return transitions
            
        elif current_state == BLEACHED_CORAL:
        
            transitions = [BLEACHED_CORAL]
            # Recovery with cool temperatures
            if temp < self.bleaching_thresh - 0.5 and healthy_neighbors > 0:
                transitions.append(HEALTHY_CORAL)
            # Death prob
            transitions.append(0)
            return transitions
            
        else:
            # Coral Creation via neighbours
            transitions = [0]
            if healthy_neighbors >= 2:
                transitions.append(HEALTHY_CORAL)
            return transitions

    def apply_biological_dynamics(self):
        
        new_coral = self.coral.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.substrate[x,y] != ROCK:
                    continue
                    
                current = self.coral[x,y]
                temp = self.temp_grid[x,y]
                neighbors = self.get_neighbors(x, y)
                healthy_neighbors = neighbors.count(HEALTHY_CORAL)
                
                #bleaching
                if current == HEALTHY_CORAL:

                    # Bleaching probability increases with DHD
                    bleach_prob = 1 - np.exp(-0.08 * self.dhd_grid[x,y])
                    if random.random() < bleach_prob:
                        new_coral[x,y] = BLEACHED_CORAL
                
                # Recovery and mortality
                elif current == BLEACHED_CORAL:
                    if temp < self.bleaching_thresh - 1.0:

                        # Recovery by healthy neighbors
                        recovery_prob = 0.15 * (1 + 0.2 * healthy_neighbors)
                        if random.random() < recovery_prob:
                            new_coral[x,y] = HEALTHY_CORAL
                    
                    # Mortality from prolonged bleaching
                    if random.random() < 0.025:
                        new_coral[x,y] = 0
                
                # New Growth
                elif current == 0:
                    if healthy_neighbors >= 2 and random.random() < 0.008:
                        new_coral[x,y] = HEALTHY_CORAL
        
        self.coral = new_coral

    def calculate_stats(self):
        #population stats
        total_rock = np.sum(self.substrate == ROCK)
        if total_rock == 0:
            return
            
        total_coral = np.sum(self.coral > 0)
        healthy = np.sum(self.coral == HEALTHY_CORAL)
        bleached = np.sum(self.coral == BLEACHED_CORAL)
        bare_rock = np.sum((self.substrate == ROCK) & (self.coral == 0))
        
        
        self.population_history['healthy'].append(healthy / total_rock)
        self.population_history['bleached'].append(bleached / total_rock)
        self.population_history['bare_rock'].append(bare_rock / total_rock)
        self.population_history['ocean'].append(np.sum(self.substrate == OCEAN))
        
        
        day_idx = min(self.current_day - 1, len(self.bleaching_data) - 1)
        target_bleached = self.bleaching_data[day_idx]
        self.population_history['target_bleached'].append(target_bleached)

    def step(self):
        
        self.update_temperature()
        
        # Alternate between biological dynamics and Potts model
        if self.current_day % 2 == 0:
            self.apply_biological_dynamics()
        else:
            self.metropolis_step()
        
        self.calculate_stats()

    def plot_reef(self, step, ax):
        
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        
        img[self.substrate == OCEAN] = [0.1, 0.3, 0.8]  # Ocean - blue
        img[(self.substrate == ROCK) & (self.coral == 0)] = [0.6, 0.6, 0.6]  # Bare rock - gray
        img[self.coral == HEALTHY_CORAL] = [0.0, 0.8, 0.2]  # Healthy coral - green
        img[self.coral == BLEACHED_CORAL] = [0.9, 0.9, 0.5]  # Bleached coral - yellow
        
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Day {self.current_day}\nTemp: {self.base_temp:.1f}°C\nBlue=Ocean, Gray=Rock, Green=Healthy, Yellow=Bleached")
        ax.axis('off')

    def plot_populations(self, ax):
        
        steps = range(len(self.population_history['healthy']))
        ax.clear()
        ax.plot(steps, self.population_history['healthy'], 'g-', label='Healthy Coral')
        ax.plot(steps, self.population_history['bleached'], 'y-', label='Simulated Bleached')
        
        
        if len(self.population_history['target_bleached']) == len(steps):
            ax.plot(steps, self.population_history['target_bleached'], 'r--', label='Target Bleached')
        
        ax.plot(steps, self.population_history['bare_rock'], 'k-', label='Bare Rock')
        ax.set_title('Population Dynamics (Proportion of Rock Area)')
        ax.set_xlabel('Day')
        ax.set_ylabel('Proportion')
        ax.legend()
        ax.grid(True)

def optimize_parameters(excel_file="Coral temps.xlsx", n_trials=20):
    
    print("Optimizing parameters")
    
    best_mse = float('inf')
    best_params = (30.0, 0.9)
    
    
    thresh_values = np.linspace(28.5, 31.5, 4)
    decay_values = np.linspace(0.85, 0.95, 4)
    
    for thresh in thresh_values:
        for decay in decay_values:
            try:
                sim = ReefSimulation(bleaching_thresh=thresh, dhd_decay=decay, excel_file=excel_file)
                
                
                for _ in range(min(len(sim.temp_data), 100)):
                    sim.step()
                
                # Calculate MSE between simulated and target bleaching
                if len(sim.population_history['bleached']) > 10:
                    simulated = np.array(sim.population_history['bleached'][-50:])
                    target = np.array(sim.population_history['target_bleached'][-50:])
                    
                    
                    valid_indices = (simulated >= 0) & (target >= 0)
                    
                    if np.sum(valid_indices) > 0:
                        mse = mean_squared_error(target[valid_indices], simulated[valid_indices])
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_params = (thresh, decay)
                            print(f"New best: thresh={thresh:.2f}, decay={decay:.2f}, MSE={mse:.4f}")
                
            except Exception as e:
                print(f"Error with params ({thresh:.2f}, {decay:.2f}): {e}")
                continue
    
    print(f"\nOptimization complete:")
    print(f"Best bleaching threshold = {best_params[0]:.2f}°C")
    print(f"Best DHD decay rate = {best_params[1]:.2f}")
    print(f"Best MSE = {best_mse:.4f}")
    return best_params

def run_animation(excel_file="Coral temps.xlsx", bleaching_thresh=30.0, dhd_decay=0.9):
    """Run animated simulation with optimized parameters"""
    print(f"\nRunning animated simulation with:")
    print(f"Bleaching threshold: {bleaching_thresh:.2f}°C")
    print(f"DHD decay rate: {dhd_decay:.2f}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    
    sim = ReefSimulation(bleaching_thresh=bleaching_thresh, dhd_decay=dhd_decay, excel_file=excel_file)
    
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    def update(frame):
        sim.step()
        
        if frame % PLOT_EVERY == 0:
            sim.plot_reef(frame, ax1)
            sim.plot_populations(ax2)
            fig.tight_layout()
        
        return ax1.images + ax2.lines if hasattr(ax1, 'images') and hasattr(ax2, 'lines') else []
    

    ani = FuncAnimation(fig, update, frames=len(sim.temp_data), interval=100, blit=False)
    plt.show()
    return ani

def run_simulation(excel_file="Coral temps.xlsx", bleaching_thresh=30.0, dhd_decay=0.9):
    
    sim = ReefSimulation(bleaching_thresh=bleaching_thresh, dhd_decay=dhd_decay, excel_file=excel_file)
    
    print(f"\nRunning simulation with:")
    print(f"Bleaching threshold: {bleaching_thresh:.2f}°C")
    print(f"DHD decay rate: {dhd_decay:.2f}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Total steps: {len(sim.temp_data)}")
    
    print("\nDay\tHealthy\tBleached\tTarget\tTemp(°C)\tDHD_max")
    print("-" * 55)
    
    for day in range(len(sim.temp_data)):
        sim.step()
        
        if day % 10 == 0 or day == len(sim.temp_data) - 1:
            temp = sim.temp_data[day]
            healthy = sim.population_history['healthy'][-1]
            bleached = sim.population_history['bleached'][-1]
            target = sim.population_history['target_bleached'][-1]
            dhd_max = np.max(sim.dhd_grid)
            
            print(f"{day:3d}\t{healthy:.3f}\t{bleached:.3f}\t\t{target:.3f}\t{temp:5.1f}\t\t{dhd_max:5.1f}")
    
    return sim

if __name__ == "__main__":
    try:
        
        best_params = optimize_parameters()
        
        # Run final simulation with optimized params
        print("\n" + "="*60)
        print("RUNNING FINAL SIMULATION")
        print("="*60)
        
        final_sim = run_simulation(
            bleaching_thresh=best_params[0], 
            dhd_decay=best_params[1]
        )
        
        # Final stats
        final_healthy = final_sim.population_history['healthy'][-1]
        final_bleached = final_sim.population_history['bleached'][-1]
        final_bare = final_sim.population_history['bare_rock'][-1]
        
        print(f"\nFinal Results (as proportion of rock area):")
        print(f"Healthy coral: {final_healthy:.3f} ({final_healthy*100:.1f}%)")
        print(f"Bleached coral: {final_bleached:.3f} ({final_bleached*100:.1f}%)")
        print(f"Bare rock: {final_bare:.3f} ({final_bare*100:.1f}%)")
        
        
        print("\n" + "="*60)
        print("RUNNING ANIMATED SIMULATION")
        print("="*60)
        
        animation = run_animation(
            bleaching_thresh=best_params[0], 
            dhd_decay=best_params[1]
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying with default parameters...")
        animation = run_animation()
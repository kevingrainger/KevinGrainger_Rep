import numpy as np
import random
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')  #needed for the kinda of animation plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
import optuna
#need to have the coral data in directory 'Coral temps.xlsx'
#params
grid_dim = 100
TOTAL_STEPS = 100
PLOT_EVERY = 5
BETA = 0.15  #beta 1/kT (interaction strenght coeff)
DHD_decay = 0.85  #degree heating days (days where the water temp is x-degrees over healthy levels)
#defining the Potts model states
OCEAN = 0
ROCK = 1
healthy_coral = 2
bleached_coral = 3
#attempt to model coral with the many state magentic system of static atoms
class ReefSimulation:
    def __init__(self, bleaching_thresh=30.0, 
                 stress_multiplier=1.0, recovery_bonus=0.3, 
                 coupling_strength=0.5, temp_sensitivity=0.8,
                 excel_file="Coral temps.xlsx"):#rather poor temperature and bleahcing data for roughly a 1km sqrt circular outcrop
        self.degree_heating_day_decay = DHD_decay
        
        #model parameters
        self.bleaching_thresh = bleaching_thresh #taken as 30 degrees 
        self.stress_multiplier = stress_multiplier #needed to change bleahcing liklihood base off DHD's or unhelathy reef
        self.recovery_bonus = recovery_bonus #the converse
        self.coupling_strength = coupling_strength #magnetism analogy
        self.temp_sensitivity = temp_sensitivity 

        #load temp data
        self.excel_file = excel_file
        self.load_temperature_data(excel_file)
        self.base_temp = np.mean(self.temp_data)

        #initialize sim
        self.current_day = 0
        reef_data = self.initialize_reef()
        self.coral = reef_data[0]
        self.substrate = reef_data[1]
        self.temp_grid = self.initialize_temperature()
        self.dhd_grid = np.zeros((grid_dim, grid_dim))
        self.recovery_days = np.zeros((grid_dim, grid_dim))# track recovery conditions
        
        #tracking for grpah
        self.population_history = {'healthy': [],'bleached': [],'bare_rock': [],'ocean': [],'target_bleached': []}

    def load_temperature_data(self, filepath):
        coral_data = pd.read_excel(filepath)
        self.temp_data = coral_data['temp'].values
        self.bleaching_data = coral_data['smooth'].values #smooth=bleahced coloumn

        if np.max(self.bleaching_data) > 1:
            self.bleaching_data = self.bleaching_data / 100.0 #data check

    def initialize_reef(self):
        coral = np.zeros((grid_dim, grid_dim), dtype=int)
        substrate = np.zeros((grid_dim, grid_dim), dtype=int)
        center = grid_dim // 2
        radius = grid_dim // 3
        for i in range(grid_dim): #there was definitely a better way to do this but i am not very fmailiar with improting spacial data
            for j in range(grid_dim): #so thereis a vaughly circular reef being generated
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                noise = 0.3 * random.random()
                if dist < radius * (0.8 + noise):
                    substrate[i,j] = ROCK
                    if random.random() < 0.75: #75% health coral cover to start
                        coral[i,j] = healthy_coral
        
        #we now simulate coral growth patterns
        for iteration in range(1000):
            x, y = random.randint(1, grid_dim-2), random.randint(1, grid_dim-2) #choos erandom gird space
            if substrate[x,y] != ROCK:
                continue
                
            #growth based on neighbors health, (loosely as its not that important in reality)
            healthy_neighbors = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                if coral[x+dx, y+dy] == healthy_coral:
                    healthy_neighbors += 1
            
            if coral[x,y] == 0 and healthy_neighbors >= 1:
                new_growth_prob = 0.02 * (healthy_neighbors / 4.0)
                if random.random() < new_growth_prob:
                    coral[x,y] = healthy_coral
            
            #small chance of coral death -> rock 
            elif coral[x,y] == healthy_coral and random.random() < 0.0005:
                coral[x,y] = 0
        
        return coral, substrate

    def initialize_temperature(self):
        #temperature grid with noise (noise keeps us out of loops and false equilibriums -hopefully)
        temp_grid = np.full((grid_dim, grid_dim), self.temp_data[0])
        variation = np.random.normal(0, 0.1, (grid_dim, grid_dim))
        return gaussian_filter(temp_grid + variation, sigma=1)

    def update_temperature(self):
        #update temperature and DHD based on current day (from excel)
        day_temp = self.temp_data[self.current_day % len(self.temp_data)]
        spatial_noise = np.random.normal(0, 0.15, (grid_dim, grid_dim))
        self.temp_grid = gaussian_filter(day_temp + spatial_noise, sigma=1.5)
        
        #calculate heat stress and decay DHD (alterable)
        heat_excess = np.maximum(self.temp_grid - self.bleaching_thresh, 0)
        
        #different decay rates for heating vs cooling (heating days tend to have a more lingering effect in regards to coral health)
        dhd_decay_rate = np.where(heat_excess > 0, 
                                 self.degree_heating_day_decay,  #normal decay
                                 0.35)  #fast decay when cooling
        
        self.dhd_grid = self.dhd_grid * dhd_decay_rate + heat_excess * 0.2
        self.dhd_grid = np.clip(self.dhd_grid, 0, 8)
        
        #track recovery days
        good_recovery_conditions = (self.temp_grid <= self.bleaching_thresh + 1.0) & (self.dhd_grid <= 2.0)
        self.recovery_days = np.where(good_recovery_conditions, 
                                     self.recovery_days + 1, 
                                     0)
        self.recovery_days = np.clip(self.recovery_days, 0, 100)
        
        self.current_day += 1
        self.base_temp = day_temp

    def get_neighbors(self, x, y):
        #get neighborhood states
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = (x + dx) % grid_dim, (y + dy) % grid_dim
            neighbors.append(self.coral[nx, ny])
        return neighbors

    def calculate_potts_energy(self, x, y, state): #note magnetisiation is a healthy reef
        #calculate energy for transition
        if self.substrate[x,y] != ROCK:
            return 0
            
        energy = 0
        temp = self.temp_grid[x,y]
        dhd = self.dhd_grid[x,y]
        recovery_days = self.recovery_days[x,y]
        neighbors = self.get_neighbors(x,y)
        healthy_neighbors = neighbors.count(healthy_coral)
        bleached_neighbors = neighbors.count(bleached_coral)
        
        #neighbor interactions (like states attract)
        for neighbor_state in neighbors:
            if neighbor_state == state:
                if state == healthy_coral:
                    energy -= 0.8 * self.coupling_strength
                elif state == bleached_coral:
                    energy -= 0.3 * self.coupling_strength
        
        if state == healthy_coral:
            #base stability for healthy coral (toward thermalization)
            energy -= 4.0
            
            #temperature stress increases energy
            if temp > self.bleaching_thresh:
                temp_stress = (temp - self.bleaching_thresh) * self.temp_sensitivity
                energy += self.stress_multiplier * temp_stress
            
            #DHD stress increases energy
            if dhd > 3.0:
                dhd_stress = np.log(1 + dhd - 3.0) * 0.5
                energy += self.stress_multiplier * dhd_stress
                    
        elif state == bleached_coral:
            #base stability for bleached
            energy -= 1.5
            recovery_energy = 0 #recovery conditions decrease energy
            if temp <= self.bleaching_thresh + 2.0 and dhd <= 4.0: # good conditions favor recovery
                recovery_energy += self.recovery_bonus * 1.5
                #timed recovery bonus
                if recovery_days > 5:
                    time_bonus = min(recovery_days / 10.0, 2.0)
                    recovery_energy += self.recovery_bonus * time_bonus
                #neighbor recovery bonus (healthy neighbours encourage regrwoth)
                if healthy_neighbors > 0:
                    neighbor_bonus = (healthy_neighbors / 4.0) * self.recovery_bonus
                    recovery_energy += neighbor_bonus
            #very good conditions strongly help recovery (had issues with bad recovery)
            if temp <= self.bleaching_thresh + 0.5 and dhd <= 1.0:
                recovery_energy += self.recovery_bonus * 2.0
                if recovery_days >= 3:
                    recovery_energy += self.recovery_bonus * 1.5
            # target recovery push (this wasneeded to get good optimisation, but i reckon if i had good training data this woulndt be needed)
            current_target = self.get_current_target()
            if current_target < 0.1:
                recovery_energy += self.recovery_bonus * 2.0
                if temp <= self.bleaching_thresh + 1.0:
                    recovery_energy += self.recovery_bonus * 1.5
            #recovery energy makes bleached less stable
            energy += recovery_energy
            # mortality under extreme conditions
            if temp > self.bleaching_thresh + 6.0 and dhd > 8.0:
                energy -= 0.5
                    
        elif state == 0:  # bare rock
            energy -= 0.5 #could probabely be lessened
            # growth potential decreases energy
            if healthy_neighbors >= 2 and temp <= self.bleaching_thresh + 1.0:
                growth_bonus = 0.3 * (healthy_neighbors / 4.0)
                energy -= growth_bonus     
        return energy

    def get_current_target(self):
        #get target bleaching for current day
        if self.current_day > 0:
            day_idx = min(self.current_day - 1, len(self.bleaching_data) - 1)
            return self.bleaching_data[day_idx]
        return 0.0

    def metropolis_step(self):
        #Metrolpolis algo, markov chain for random tranistions, will decide to accept transition by an energy consideration
        new_coral = self.coral.copy()
        attempts_per_step = int(grid_dim * grid_dim * 0.8) #was too slow 
        
        for _ in range(attempts_per_step):
            x, y = random.randint(0, grid_dim-1), random.randint(0, grid_dim-1)#choose spot
            if self.substrate[x,y] != ROCK:
                continue
                
            current_state = self.coral[x,y]
            possible_states = self.get_allowed_transitions(current_state, x, y)
            if len(possible_states) <= 1: #toubleshooting
                continue  
            new_state = random.choice(possible_states)
            if new_state == current_state:
                continue
            #calculate energy difference
            old_energy = self.calculate_potts_energy(x, y, current_state)
            new_energy = self.calculate_potts_energy(x, y, new_state)
            delta_E = new_energy - old_energy
            # recovery bias when target is low (again with better trainging data this approach could be gotten rid of)
            if current_state == bleached_coral and new_state == healthy_coral:
                current_target = self.get_current_target()
                temp = self.temp_grid[x,y]
                dhd = self.dhd_grid[x,y]
                
                if current_target < 0.1:
                    delta_E -= 0.8  #strong recovery bias
                    
                    if temp <= self.bleaching_thresh + 1.0 and dhd <= 2.0:
                        delta_E -= 0.5
                
                #general recovery bias
                elif temp <= self.bleaching_thresh + 2.0 and dhd <= 4.0:
                    delta_E -= 0.3
            
            #add small random noise (stop loops)
            delta_E += np.random.normal(0, 0.02)
            #Metropolis acceptance crit
            if delta_E <= 0:
                new_coral[x,y] = new_state
            else:
                acceptance_prob = np.exp(-BETA * delta_E)
                if random.random() < acceptance_prob:
                    new_coral[x,y] = new_state
        
        self.coral = new_coral

    def get_allowed_transitions(self, current_state, x, y): #(easier this way as I had to add alot of conidtions in previous func)
        transitions = [current_state]
        temp = self.temp_grid[x,y]
        dhd = self.dhd_grid[x,y]
        recovery_days = self.recovery_days[x,y]
        
        if current_state == healthy_coral:
            #allow bleaching under stress
            if temp > self.bleaching_thresh or dhd > 2.0:
                transitions.append(bleached_coral)
            # direct mortality under extreme conditions
            if temp > self.bleaching_thresh + 8.0 and dhd > 10.0:
                transitions.append(0)
            
        elif current_state == bleached_coral:
            #always allow recovery
            transitions.append(healthy_coral)
            #mortality under severe stress
            if dhd > 8.0 and temp > self.bleaching_thresh + 6.0:
                transitions.append(0)
                
        elif current_state == 0:
            #growth with healthy neighbors
            healthy_neighbors = self.get_neighbors(x, y).count(healthy_coral)
            if healthy_neighbors >= 2 and temp <= self.bleaching_thresh + 1.0:
                if random.random() < 0.1:
                    transitions.append(healthy_coral)
                
        return transitions

    def step(self):
        if self.current_day >= len(self.temp_data):
            return False
        
        self.update_temperature()
        self.metropolis_step()
        self.calculate_stats()
        return True

    def calculate_stats(self): #for visuals
        total_rock = np.sum(self.substrate == ROCK)
        if total_rock == 0:
            return
            
        total_coral = np.sum(self.coral > 0)
        healthy = np.sum(self.coral == healthy_coral)
        bleached = np.sum(self.coral == bleached_coral)
        bare_rock = np.sum((self.substrate == ROCK) & (self.coral == 0))
        
        self.population_history['healthy'].append(healthy / total_rock)
        self.population_history['bleached'].append(bleached / total_rock)
        self.population_history['bare_rock'].append(bare_rock / total_rock)
        self.population_history['ocean'].append(np.sum(self.substrate == OCEAN))
        
        #store target bleaching
        day_idx = min(self.current_day - 1, len(self.bleaching_data) - 1)
        target_bleached = self.bleaching_data[day_idx]
        self.population_history['target_bleached'].append(target_bleached)

    def plot_reef(self, step, ax):
        img = np.zeros((grid_dim, grid_dim, 3))
        #color mapping
        img[self.substrate == OCEAN] = [0.1, 0.3, 0.8]
        img[(self.substrate == ROCK) & (self.coral == 0)] = [0.6, 0.6, 0.6]
        img[self.coral == healthy_coral] = [0.0, 0.8, 0.2]
        img[self.coral == bleached_coral] = [0.9, 0.9, 0.5]
        
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Day {self.current_day}\nTemp: {self.base_temp:.1f}°C\nBlue=Ocean, Gray=Rock, Green=Healthy, Yellow=Bleached")
        ax.axis('off')

    def plot_populations(self, ax): #plot population over time
        steps = range(len(self.population_history['healthy']))
        ax.clear()
        ax.plot(steps, self.population_history['healthy'], 'g-', label='Healthy Coral', linewidth=2)
        ax.plot(steps, self.population_history['bleached'], 'y-', label='Simulated Bleached', linewidth=2)
        
        #plot target
        if len(self.population_history['target_bleached']) == len(steps):
            ax.plot(steps, self.population_history['target_bleached'], 'r--', label='Target Bleached', linewidth=2)
        
        ax.plot(steps, self.population_history['bare_rock'], 'k-', label='Bare Rock', alpha=0.7)
        ax.set_title('Population Dynamics (Proportion of Rock Area)')
        ax.set_xlabel('Day')
        ax.set_ylabel('Proportion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

def optimize_parameters(excel_file="Coral temps.xlsx", n_trials=20):
    #optimize model parameters using Optuna (imported, not sure if it is any good, not really familar)
    def objective(trial):
        #parameter ranges (quickens it all)
        stress_multiplier = trial.suggest_float("stress_mult", 0.5, 1.5)
        recovery_bonus = trial.suggest_float("recovery", 0.3, 1.0)
        coupling_strength = trial.suggest_float("coupling", 0.2, 0.8)
        temp_sensitivity = trial.suggest_float("temp_sens", 0.5, 1.2)
        sim = ReefSimulation(
            stress_multiplier=stress_multiplier,
            recovery_bonus=recovery_bonus,
            coupling_strength=coupling_strength,
            temp_sensitivity=temp_sensitivity,
            excel_file=excel_file
        )

        #run simulation and calculat error
        errors = []
        max_days = min(len(sim.temp_data), 160)
        for day in range(max_days):
            if not sim.step():
                break
            
            if day >= 10 and day % 5 == 0:
                bleached_sim = sim.population_history['bleached'][-1]
                target = sim.population_history['target_bleached'][-1]
                error = (bleached_sim - target)**2
                errors.append(error)
                
                #stopping if diverging
                if len(errors) > 10:
                    recent_avg = np.mean(errors[-3:])
                    if recent_avg > 0.5:
                        break

        if len(errors) == 0:
            final_error = 5.0
        else:
            final_error = np.mean(errors)
        
        #penalize poor recovery (again too strong handed a measure)
        if len(sim.population_history['target_bleached']) > 50:
            last_20_targets = sim.population_history['target_bleached'][-20:]
            last_20_simulated = sim.population_history['bleached'][-20:]
            
            low_target_indices = [i for i, t in enumerate(last_20_targets) if t < 0.1]
            if len(low_target_indices) > 10:
                avg_sim_during_low = np.mean([last_20_simulated[i] for i in low_target_indices])
                if avg_sim_during_low > 0.2:
                    final_error += 2.0
        
        return final_error

    #run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value:.3f}")
    print(f"Best objective value: {study.best_value:.4f}")
    
    return {
        'stress_multiplier': study.best_params['stress_mult'],
        'recovery_bonus': study.best_params['recovery'],
        'coupling_strength': study.best_params['coupling'],
        'temp_sensitivity': study.best_params['temp_sens']
    }

def run_animation(excel_file="Coral temps.xlsx", bleaching_thresh=30.0,
                 stress_multiplier=1.0, recovery_bonus=0.3, 
                 coupling_strength=0.5, temp_sensitivity=0.8):
    print(f"\nRunning sim...")
    print(f"Grid size: {grid_dim}x{grid_dim}")
    print(f"Beta (temperature): {BETA}")
    
    sim = ReefSimulation(
        bleaching_thresh=bleaching_thresh,
        stress_multiplier=stress_multiplier,
        recovery_bonus=recovery_bonus,
        coupling_strength=coupling_strength,
        temp_sensitivity=temp_sensitivity,
        excel_file=excel_file
    )
    
    #print initial cond.
    total_rock = np.sum(sim.substrate == ROCK)
    initial_bare = np.sum((sim.substrate == ROCK) & (sim.coral == 0))
    print(f"Initial bare rock: {initial_bare/total_rock*100:.1f}% of rock area")
    
    #figure
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    def update(frame):
        #animation update function (not good at this stuff)
        if not sim.step():
            return []
        if frame % PLOT_EVERY == 0:
            sim.plot_reef(frame, ax1)
            sim.plot_populations(ax2)
            fig.tight_layout()
        
        return []
    
    ani = FuncAnimation(fig, update, frames=len(sim.temp_data), interval=150, blit=False)
    return ani

def run_simulation(excel_file="Coral temps.xlsx", bleaching_thresh=30.0,
                  stress_multiplier=1.0, recovery_bonus=0.3, 
                  coupling_strength=0.5, temp_sensitivity=0.8):
    #run standard simulation (I should merger these there is no need for 2 funcitons, I was just having trouble with the animation)
    sim = ReefSimulation(bleaching_thresh=bleaching_thresh,
                        stress_multiplier=stress_multiplier, recovery_bonus=recovery_bonus,
                        coupling_strength=coupling_strength, temp_sensitivity=temp_sensitivity, 
                        excel_file=excel_file)

    total_rock = np.sum(sim.substrate == ROCK)
    initial_bare = np.sum((sim.substrate == ROCK) & (sim.coral == 0))
    initial_healthy = np.sum(sim.coral == healthy_coral)
    print(f"\nInitial conditions:")
    print(f"Initial bare rock: {initial_bare/total_rock*100:.1f}% of rock area")
    print(f"Initial healthy coral: {initial_healthy/total_rock*100:.1f}% of rock area")
    print("\nDay\tHealthy\tBleached\tTarget\tTemp\tDHD_max\tRecovery_days_max")

    
    for day in range(len(sim.temp_data)):
        if not sim.step():
            break
        
        if day % 10 == 0 or day == len(sim.temp_data) - 1:
            temp = sim.temp_data[day]
            healthy = sim.population_history['healthy'][-1]
            bleached = sim.population_history['bleached'][-1]
            target = sim.population_history['target_bleached'][-1]
            dhd_max = np.max(sim.dhd_grid)
            recovery_days_max = np.max(sim.recovery_days)
            
            print(f"{day:3d}\t{healthy:.3f}\t{bleached:.3f}\t\t{target:.3f}\t{temp:5.1f}\t\t{dhd_max:5.1f}\t\t{recovery_days_max:5.0f}")
    
    return sim

if __name__ == "__main__":
    #random seeds (needed to meaningfuly optimise)
    random.seed(42)
    np.random.seed(42)
    
    #run with optimization
    print("Starting Optimisation of parameters...")
    best_params = optimize_parameters()
    best_params['bleaching_thresh'] = 30.0
    final_sim = run_simulation(**best_params)
    #final results
    final_healthy = final_sim.population_history['healthy'][-1]
    final_bleached = final_sim.population_history['bleached'][-1]
    final_bare = final_sim.population_history['bare_rock'][-1]
    
    print(f"\nFinal Results:")
    print(f"Healthy coral: {final_healthy:.3f} ({final_healthy*100:.1f}%)")
    print(f"Bleached coral: {final_bleached:.3f} ({final_bleached*100:.1f}%)")
    print(f"Bare rock: {final_bare:.3f} ({final_bare*100:.1f}%)")
    #run animation
    print("\n" + "="*60)
    animation = run_animation(**best_params)
    plt.show(block=True)
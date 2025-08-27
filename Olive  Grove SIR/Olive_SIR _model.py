import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Tuple, List, Dict
import random
import time
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

class CellType(Enum):
    SHRUB = 0
    TREE = 1
    EMPTY = 2

class TreeState(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class SproutState(Enum):
    TENDER = 0
    HARD = 1


##############--------------------------------------
class OliveGrove:

    #first defining a function to setup the orchard shape
    def _plant_trees(self):
        self.tree_positions = []
            
        #simple grid planting
        for y in range(self.tree_spacing, self.height - self.tree_spacing, self.tree_spacing):
            for x in range(self.tree_spacing, self.width - self.tree_spacing, self.tree_spacing):
                # Check space
                if y + self.tree_size < self.height and x + self.tree_size < self.width:
                    # Plant tree
                    for tree_y in range(y, y + self.tree_size):
                        for tree_x in range(x, x + self.tree_size):
                            self.cell_type[tree_y, tree_x] = CellType.TREE.value
                        
                    #tree center position (needed later for force dynamics etc.)
                    center_y = y + self.tree_size // 2
                    center_x = x + self.tree_size // 2
                    tree_center = (center_y, center_x)
                        
                    self.tree_health[tree_center] = TreeState.SUSCEPTIBLE
                    self.tree_positions.append(tree_center)


#################------------------------------------

    def __init__(self, width: int = 90, height: int = 90, tree_spacing: int = 5, tree_size: int = 3, vectors_per_hectare: int = 1000000, sampling_rate: float = 1.0):  
        #Default to no sampling (but mine doesn't run without it)
        #setting defaults
        self.width = width
        self.height = height
        self.tree_spacing = tree_spacing
        self.tree_size = tree_size
        self.sampling_rate = sampling_rate
        
        #Initialize grids
        self.cell_type = np.full((height, width), CellType.SHRUB.value, dtype=np.uint8) #made all SHRUB 
        self.tree_health = {}
        self.infection_time = {}
        self.symptom_time = {}
        #Plant trees
        self._plant_trees() #using beginning tree funciton
        #paper uses 1M vectors/hectare but my little computer can't do all that
        area_hectares = (width * height) / 10000
        self.actual_vectors = int(vectors_per_hectare * area_hectares)#vectors being the bugs
        self.simulated_vectors = int(self.actual_vectors * sampling_rate) #needed sampling for it to run, it definitely is not as accurate
        
        self.vector_x = np.zeros(self.simulated_vectors, dtype=np.int16)#positions
        self.vector_y = np.zeros(self.simulated_vectors, dtype=np.int16)
        self.vector_infected = np.zeros(self.simulated_vectors, dtype=bool) #state of vecotrs
        self.vector_alive = np.ones(self.simulated_vectors, dtype=bool)
        
        print(f"Using {self.simulated_vectors:,} vectors")
        print(f"(representing {self.actual_vectors:,} actual vectors with {sampling_rate*100:.1f}% sampling)")
        print(f"Area of grove: {area_hectares:.2f} hectares")#what was used in paper
        
        #time setup
        self.time_step = 0
        self.day = 0
        self.time_steps_per_day = 48  #30 min intervals, all these are informed by the paper
        self.sprout_state = SproutState.TENDER
        self.hardening_day = 185  #day when olive twigs harden
        #Movement probabilities from paper, change based on twig states
        self.p_tt = 0.3  # twig to twig (reduced for capacity)
        self.p_bb = 0.15  # branch to branch
        self.p_ss = 0.5  # SHRUB to SHRUB
        #-------variable probs---------
        self.current_p_ts = 0.005  # tree to SHRUB (tender)
        self.current_p_st = 1.0    # SHRUB to tree (tender)
        #transmission parameters from paper
        self.tree_susceptibility = 1.0  
        self.vector_susceptibility = 1.0  
        
        if sampling_rate < 1.0: #
            # If sampling, use modest compensation
            self.tranmission_for_sampled_case = min(1.5, 1.0 / sampling_rate ** 0.5)
            self.tree_susceptibility = min(1.0, self.tree_susceptibility * self.tranmission_for_sampled_case)
        
        #propagation of xylems (from paper), and symptoms
        self.xylem_propagation_rate = 0.167  # cm/day
        self.twig_length = 15  #cm
        self.symptom_delay_mean = 730  #days , I presumje the paper has a source for this
        self.symptom_delay_std = 100   #days
        #stats
        self.infection_history = []
        self.vector_infection_history = []
        self.daily_new_infections = 0
        
####################------------------------------------

    def initialize_vectors(self, initial_infected: int = 1):
        
        SHRUB_positions = np.argwhere(self.cell_type == CellType.SHRUB.value) #find all shrub positions
        if len(SHRUB_positions) > 0:
            #randomly place vectors on SHRUBs
            selected_vector_pos = np.random.choice(
                len(SHRUB_positions), 
                size=min(self.simulated_vectors, len(SHRUB_positions)), 
                replace=True #allow more than one vector per shrub
            )
            selected_positions = SHRUB_positions[selected_vector_pos]
            
            for i in range(len(selected_positions)):
                y, x = selected_positions[i]  #get coordinates
                self.vector_y[i] = y          
                self.vector_x[i] = x    
        #initial infected vectors
        if initial_infected > 0:
            infected_vector_indices = np.random.choice(self.simulated_vectors,size=min(initial_infected, self.simulated_vectors),                              replace=False)
            self.vector_infected[infected_vector_indices] = True
    
###################--------------------------------

    def update_growth_phase(self):
        
        day_of_year = self.day % 365

        #paper says tender from day 90 to 185 - hard otherwise
        if 90 <= day_of_year <= 185:  # ← HARD CODE 185 HERE
            if self.sprout_state != SproutState.TENDER:
                print(f"Day {self.day}: Switched to TENDER sprouts")
            self.sprout_state = SproutState.TENDER
            
            #vectors prefer trees when sprouts are tender
            self.current_p_ts = 0.005  # Low probability to leave trees
            self.current_p_st = 1.0    # High probability to move to trees
        else:
            if self.sprout_state != SproutState.HARD:
                print(f"Day {self.day}: Switching to HARD sprouts")
            self.sprout_state = SproutState.HARD
            
            #vectors are forced to shrubs when sprouts are hard  
            self.current_p_ts = 1.0  # MUST leave trees
            self.current_p_st = 0.0  # CANNOT move to trees

##############--------------------------------------------------
    def _get_tree_center(self, x: int, y: int) -> Tuple[int, int]:
        
        if self.cell_type[y, x] != CellType.TREE.value:
            return None
        
        for cy, cx in self.tree_positions:
            if abs(cy - y) <= 1 and abs(cx - x) <= 1:
                return (cy, cx)
        return None

##############-------------------------------------------------

    def vector_movement(self): #this was hard
        
        vector_life_status_bool = self.vector_alive
        
        for i in np.where(vector_life_status_bool)[0]:
            x = self.vector_x[i]
            y = self.vector_y[i]
            current_cell = self.cell_type[y, x]
            
            #symptomatic trees repel vectors (defined in the function below)
            if current_cell == CellType.TREE.value:
                tree_center = self._get_tree_center(x, y)
                if tree_center and self.tree_health.get(tree_center) == TreeState.REMOVED:
                    self._repulsion_from_symptomatic_tree(i)
                    continue
            
            #random movement decision, brownain motion ish
            if random.random() < 0.5:  #attempt movement
                #random direction
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dx, dy = random.choice(directions)
                new_x, new_y = x + dx, y + dy
                
                #check boundaries
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    new_cell = self.cell_type[new_y, new_x]
                    #check if new position is symptomatic tree
                    if new_cell == CellType.TREE.value:
                        tree_center = self._get_tree_center(new_x, new_y)
                        if tree_center and self.tree_health.get(tree_center) == TreeState.REMOVED:
                            continue  #don't move to symptomatic trees
                    
                    move = False #start with no movement
                    if current_cell == CellType.SHRUB.value: #i kept gettign crashes from empty lists
                        if new_cell == CellType.SHRUB.value:
                            move = random.random() < self.p_ss #defined in main function, maybe would make more sense here
                        elif new_cell == CellType.TREE.value:
                            move = random.random() < self.current_p_st
                    elif current_cell == CellType.TREE.value:
                        if new_cell == CellType.TREE.value:
                            #check if same tree or different
                            if abs(new_x - x) <= 1 and abs(new_y - y) <= 1:
                                move = random.random() < self.p_tt
                            else:
                                move = random.random() < self.p_bb
                        elif new_cell == CellType.SHRUB.value:
                            move = random.random() < self.current_p_ts
                    
                    if move:
                        self.vector_x[i] = new_x
                        self.vector_y[i] = new_y

###################--------------------------------
    def _repulsion_from_symptomatic_tree(self, vector_idx):
       
        x = self.vector_x[vector_idx]
        y = self.vector_y[vector_idx]
        #find nearby SHRUB cells
        for dx in [-1, 0, 1]: #8 in neighbourhood
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue #skip current
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.cell_type[new_y, new_x] == CellType.SHRUB.value): #check boundaries
                    self.vector_x[vector_idx] = new_x #pushing vector
                    self.vector_y[vector_idx] = new_y
                    return
                
###################-------------------------------------
    def transmission(self):
       
        self.daily_new_infections = 0 #for tracking
        vector_life_status_bool = self.vector_alive
        
        for i in np.where(vector_life_status_bool)[0]:
            x = self.vector_x[i]
            y = self.vector_y[i]
            
            #only transmission on trees
            if self.cell_type[y, x] != CellType.TREE.value:
                continue
            
            tree_center = self._get_tree_center(x, y)
            if tree_center is None:
                continue #if removed
            
            tree_state = self.tree_health[tree_center]
            #skip symptomatic trees
            if tree_state == TreeState.REMOVED:
                continue
            
            #Infected vector to Susceptible tree
            if self.vector_infected[i] and tree_state == TreeState.SUSCEPTIBLE:
                #stochastic transmission with tree susceptibility
                if random.random() < self.tree_susceptibility: #set prior
                    self.tree_health[tree_center] = TreeState.INFECTED
                    self.infection_time[tree_center] = self.day
                    
                    #symptom appearance time (stochastic also)
                    symptom_delay = np.random.normal(self.symptom_delay_mean, self.symptom_delay_std)
                    self.symptom_time[tree_center] = self.day + max(0, symptom_delay)
                    self.daily_new_infections += 1
            
            #susceptible vector infected by infected tree
            elif not self.vector_infected[i] and tree_state == TreeState.INFECTED:
                days_infected = self.day - self.infection_time.get(tree_center, 0)
                
                #calculate infection level (trees become more infectious)
                if days_infected < 90:  # 3 months of infection
                    colonization = min(1.0, (days_infected * self.xylem_propagation_rate) / self.twig_length)
                    acquisition_prob = self.vector_susceptibility * colonization
                else:
                    acquisition_prob = self.vector_susceptibility
                
                #stochastic acquisition again
                if random.random() < acquisition_prob:
                    self.vector_infected[i] = True

###################-------------------------------------------
    def update_disease_state(self):
       
        for tree_center in list(self.tree_health.keys()):
            if self.tree_health[tree_center] == TreeState.INFECTED:
                if tree_center in self.symptom_time and self.day >= self.symptom_time[tree_center]:
                    self.tree_health[tree_center] = TreeState.REMOVED
                    #symptomatic trees no longer participate after given time, removed/died#

######################--------------------------------------------
    def step(self):

        self.time_step += 1
        #daily updates
        if self.time_step % self.time_steps_per_day == 0:
            self.day += 1
            self.update_growth_phase()
            self.update_disease_state()
            self._record_statistics()
            #annual vector reset (winter die-off)
            if self.day % 365 == 120:
                self._annual_vector_reset()
        #daily processes
        self.vector_movement()
        self.transmission()

#################------------------------------

    def _annual_vector_reset(self):
        print(f"Day {self.day}: Annual vector reset")
        
        #reset ALL vectors to susceptible 
        self.vector_alive.fill(True)
        self.vector_infected.fill(False)  #all start susceptible
        
        #redistribute on shrubs - no initially infected vectors
        self.initialize_vectors(initial_infected=0)
        
###############-----------------------

    def _record_statistics(self):
        #tree statistics
        states = list(self.tree_health.values())
        total_trees = len(states)
        if total_trees > 0:
            s_count = states.count(TreeState.SUSCEPTIBLE)
            i_count = states.count(TreeState.INFECTED)
            r_count = states.count(TreeState.REMOVED)
            
            self.infection_history.append({
                'day': self.day,
                'S': s_count / total_trees,
                'I': i_count / total_trees,
                'R': r_count / total_trees,
                'new_infections': self.daily_new_infections
            })
        
        #vector statistics
        vector_life_status_bool = self.vector_alive
        if np.any(vector_life_status_bool):
            infected_fraction = np.mean(self.vector_infected[vector_life_status_bool])
            self.vector_infection_history.append(infected_fraction)

################-----------------  

    def visualize(self, show_vectors=True, show_statistics=True):
        sns.set_style("whitegrid")
        
        if show_statistics: #again to stop crashing from empty lists, prolly don tneed this one
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[:, :2])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[1, 2])
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        
        
        display_grid = np.zeros((self.height, self.width, 3))#main orchard visualization
        #shrubs
        SHRUB_bool = self.cell_type == CellType.SHRUB.value #boolean 'mask'
        display_grid[SHRUB_bool] = [0.85, 0.95, 0.85] #rgb needed for matplotlib??
        
        #trees
        for tree_center, state in self.tree_health.items():
            cy, cx = tree_center #as defined before 
            y_slice = slice(max(0, cy-1), min(self.height, cy+2)) #finding a 3x3 block of the tree
            x_slice = slice(max(0, cx-1), min(self.width, cx+2))#min and max stop out of bounds errors
            
            if state == TreeState.SUSCEPTIBLE:
                display_grid[y_slice, x_slice] = [0.2, 0.6, 0.2] #colours for different tree states
            elif state == TreeState.INFECTED:
                display_grid[y_slice, x_slice] = [0.9, 0.9, 0.3]
            else:  # REMOVED
                display_grid[y_slice, x_slice] = [0.8, 0.2, 0.2]
        
        ax1.imshow(display_grid, interpolation='nearest')
        
        
        #main olive grove chart setup
        ax1.set_title(f'Orchard (Day {self.day}) - {len(self.tree_health)} trees\n',fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        
        #add vectors
        if show_vectors and np.sum(self.vector_alive) > 0:
            vector_life_status_bool = self.vector_alive
            infected_vector_bool = self.vector_infected & vector_life_status_bool #alive vector required
            healthy_vector_bool = ~self.vector_infected & vector_life_status_bool
            
            #show subset of vectors
            max_display = min(2000, np.sum(vector_life_status_bool))
            
            if np.any(healthy_vector_bool):
                healthy_idx = np.where(healthy_vector_bool)[0][:max_display//2] #short hand
                ax1.scatter(self.vector_x[healthy_idx], self.vector_y[healthy_idx],
                          c='blue', s=3, alpha=0.5, label='Healthy vectors') #blue for helathy vectors
            
            if np.any(infected_vector_bool):
                infected_idx = np.where(infected_vector_bool)[0][:max_display//2]
                ax1.scatter(self.vector_x[infected_idx], self.vector_y[infected_idx],
                          c='red', s=3, alpha=0.7, label='Infected vectors') #red for infected
        
        #tree legend with rgb colours
        legend_elements = [
            patches.Patch(color=[0.2, 0.6, 0.2], label='Susceptible'),
            patches.Patch(color=[0.9, 0.9, 0.3], label='Infected'),
            patches.Patch(color=[0.8, 0.2, 0.2], label='Symptomatic'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        if show_statistics and self.infection_history: #check for none empy list.
            #SIR with seasonal patterns
            days = [d['day'] for d in self.infection_history]
            S = [d['S'] for d in self.infection_history]
            I = [d['I'] for d in self.infection_history]
            R = [d['R'] for d in self.infection_history]
            
            ax2.plot(days, S, 'g-', label='Susceptible', linewidth=2)
            ax2.plot(days, I, 'y-', label='Infected', linewidth=2)
            ax2.plot(days, R, 'r-', label='Removed', linewidth=2) #removed rather than recovered as per paper, and obvs as the tree dies
            
            #mark growth phase transitions with backgorund shading
            for year in range(int(max(days)/365) + 1):
                ax2.axvspan(year*365 + 90, year*365 + 185, alpha=0.1, color='green', label='Tender' if year==0 else '')
                ax2.axvspan(year*365 + 185, year*365 + 365, alpha=0.1, color='brown', label='Hard' if year==0 else '')
                ax2.axvspan(year*365, year*365 + 90, alpha=0.1, color='brown')
            
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Fraction of trees')
            ax2.set_title('SIR with Seasonal Phases', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            #vector infection graph, final graph bottom right
            if self.vector_infection_history:
                ax3.plot(self.vector_infection_history, 'r-', linewidth=2)
                ax3.fill_between(range(len(self.vector_infection_history)),self.vector_infection_history,alpha=0.3, color='red')
                ax3.set_xlabel('Days')
                ax3.set_ylabel('Fraction infected')
                ax3.set_title('Vector Infection Rate', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        return fig
    

##################################-------------------------------

def run_paper_simulation(days=365*3, tree_spacing=5, sampling_rate=0.05, time_steps_per_day=48):
    #loading params as per the paper
    orchard = OliveGrove(
        width=90,
        height=90,
        tree_spacing=tree_spacing,
        vectors_per_hectare=1000000,
        sampling_rate=sampling_rate 
    )
    
    orchard.time_steps_per_day = time_steps_per_day #thinking i did this weird
    orchard.initialize_vectors(initial_infected=1)
    
    #just in case i need to change them 
    print(f"Orchard: {orchard.width}m × {orchard.height}m")
    print(f"Trees: {len(orchard.tree_health)}")
    print(f"Vectors: {orchard.simulated_vectors:,}")
    print(f"Sampling rate: {sampling_rate*100:.1f}%")
    print(f"Time steps/day: {orchard.time_steps_per_day}")
    
    start_time = time.time() #set clock
    for day in range(days):
        for _ in range(orchard.time_steps_per_day):
            orchard.step()
        
        if day % 30 == 0 and day > 0:
            if orchard.infection_history: #a fix for crashes if the there are no infections
                current = orchard.infection_history[-1]
                print(f"Day {day}: S={current['S']:.1%}, I={current['I']:.1%}, "
                      f"R={current['R']:.1%} |")
    
    #statistics
    total_time = time.time() - start_time
    if orchard.infection_history:
        final = orchard.infection_history[-1]
        print(f"Final state: S={final['S']:.1%}, I={final['I']:.1%}, R={final['R']:.1%}")
    #fig- i should make it an animation
    fig = orchard.visualize(show_vectors=True, show_statistics=True)
    plt.show()
    
    return orchard

#####################------------------------------

if __name__ == "__main__":
    # Run for 3 years as in paper
    orchard = run_paper_simulation(
        days=365*3, 
        tree_spacing=5, #this was altered in the paper to find optimal spacing
        sampling_rate=0.001,  # 1% sampling
        time_steps_per_day=30  #Paper uses 48 30min steps
    )
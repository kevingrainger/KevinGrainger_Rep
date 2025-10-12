import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random
import time

#Cell types
CELL_TYPES = {'shrub': 0, 'tree': 1, 'empty': 2}
#Tree health states
TREE_STATES = {'susceptible': 0, 'infected': 1, 'removed': 2}
#Sprout states
SPROUT_STATES = {'tender': 0, 'hard': 1}

class OliveGrove:

    def _plant_trees(self): #this was weirdly hard
            self.tree_positions = []
            #simple grid planting
            for y in range(self.tree_spacing, self.height - self.tree_spacing, self.tree_spacing): #start, stop, step
                for x in range(self.tree_spacing, self.width - self.tree_spacing, self.tree_spacing):
                    #Check space
                    if y + self.tree_size < self.height and x + self.tree_size < self.width:
                        #Plant tree
                        for tree_y in range(y, y + self.tree_size):
                            for tree_x in range(x, x + self.tree_size):
                                self.cell_type[tree_y, tree_x] = CELL_TYPES['tree']
                        #tree center position (needed later for force dynamics etc.)
                        center_y = y + self.tree_size // 2
                        center_x = x + self.tree_size // 2
                        tree_center = (center_y, center_x)
                        self.tree_health[tree_center] = TREE_STATES['susceptible']
                        self.tree_positions.append(tree_center)

    def __init__(self, width=90, height=90, tree_spacing=5, tree_size=3, 
                 vectors_per_hectare=1000000, sampling_rate=1.0):
    #-------------- defining a function to setup the Olive grove shape-----------------
        

        #---------Basic Olive Grove Setup-------------------
        #Paper did no sampling (but mine doesn't run without it)
        self.width = width
        self.height = height
        self.tree_spacing = tree_spacing
        self.tree_size = tree_size
        self.sampling_rate = sampling_rate 
        self.cell_type = np.full((height, width), CELL_TYPES['shrub'], dtype=np.uint8) #made all SHRUB,  Initialize grids
        self.tree_health = {} #starting needed lists for graph
        self.infection_time = {}
        self.symptom_time = {}
        self._plant_trees() #Plant trees using beginning tree funciton
        #paper uses 1M vectors/hectare but my little computer can't do that
        area_hectares = (width * height) / 10000
        self.actual_vectors = int(vectors_per_hectare * area_hectares)#vectors being the bugs
        self.simulated_vectors = int(self.actual_vectors * sampling_rate) #needed sampling for it to run, it definitely is not as accurate. sampling just allows me to scale up or down easier
        #-----------------Defining Vector positions and states-----------------
        self.vector_x = np.zeros(self.simulated_vectors, dtype=np.int16)#positions
        self.vector_y = np.zeros(self.simulated_vectors, dtype=np.int16)
        self.vector_infected = np.zeros(self.simulated_vectors, dtype=bool) #state of vecotrs
        self.vector_alive = np.ones(self.simulated_vectors, dtype=bool)
        print(f"Using {self.simulated_vectors:,} vectors")
        
        #----------------time setup--------------------------------
        self.time_step = 0
        self.day = 0
        self.time_steps_per_day = 48  #30 min intervals, all these are informed by the original paper
        self.sprout_state = SPROUT_STATES['tender'] #start as tender sprouts
        self.hardening_day = 185  #day when olive twigs harden
        #----------Movement probabilities from paper (change based on twig states)--------------
        self.p_tt = 0.3  # twig to twig (reduced for capacity)
        self.p_bb = 0.15  # branch to branch
        self.p_ss = 0.5  # SHRUB to SHRUB
        #------------variable probabilities-----------------------------
        self.current_p_ts = 0.005  # tree to SHRUB (tender)
        self.current_p_st = 1.0    # SHRUB to tree (tender)
        #-------------transmission parameters from paper------------------
        self.tree_susceptibility = 1.0  
        self.vector_susceptibility = 1.0  
        #--------------Correcting due ot  (boost infection rates)---------------------
        if sampling_rate < 1.0: 
            # If sampling, use modest compensation to make up for lack of vectors
            compensation_factor = 1.0 / (sampling_rate ** 0.5) #root
            self.tranmission_for_sampled_case = min(1.5, compensation_factor) #caps the boost
            adjusted_susceptibility = self.tree_susceptibility * self.tranmission_for_sampled_case
            self.tree_susceptibility = min(1.0, adjusted_susceptibility) #cap, this fixed my code
        
        #-------------propagation of xylems (from paper), and symptoms----------------
        self.xylem_propagation_rate = 0.167  # cm/day
        self.twig_length = 15  #cm
        self.symptom_delay_mean = 730  #days , I presumje the paper has a source for this
        self.symptom_delay_std = 100   #days
        #-------------------stats-----------------------
        self.infection_history = []
        self.vector_infection_history = []
        self.daily_new_infections = 0

    #---------------------Starting the vectors off---------------------
    def initialize_vectors(self, initial_infected=1):
        SHRUB_positions = np.argwhere(self.cell_type == CELL_TYPES['shrub']) #find all shrub positions given we planted tree, we could also just do excl. trees

        if len(SHRUB_positions) > 0:
            #randomly place vectors on SHRUBs
            num_positions = len(SHRUB_positions)
            num_vectors = min(self.simulated_vectors, num_positions) #caps incase too manyt vecotrs exisit, this was an earlier issue i could delete this
            
            selected_vector_pos = np.random.choice(num_positions, size=num_vectors, replace=True) #allow more than one vector per shrub
            selected_positions = SHRUB_positions[selected_vector_pos]
            
            for i in range(len(selected_positions)):
                y = selected_positions[i][0]  #get coordinates
                x = selected_positions[i][1]
                self.vector_y[i] = y          
                self.vector_x[i] = x
        
        #initial infected vectors
        if initial_infected > 0:
            num_to_infect = min(initial_infected, self.simulated_vectors)
            infected_vector_indices = np.random.choice(self.simulated_vectors, size=num_to_infect, replace=False)#replace=false so you cant double count infected planta
            self.vector_infected[infected_vector_indices] = True

#--------------Keeping track of the growth phases------------------
    def update_growth_phase(self):
        day_of_year = self.day % 365
        #paper says tender from day 90 to 185 - hard otherwise
        if 90 <= day_of_year <= 185:
            if self.sprout_state != SPROUT_STATES['tender']: #print the day it changes
                print(f"Day {self.day}: Switched to tender sprouts")
            self.sprout_state = SPROUT_STATES['tender']
            
            #vectors prefer trees when sprouts are tender
            self.current_p_ts = 0.005  # Low probability to leave trees
            self.current_p_st = 1.0    # High probability to move to trees
        else:
            if self.sprout_state != SPROUT_STATES['hard']:
                print(f"Day {self.day}: Switching to hard sprouts")
            self.sprout_state = SPROUT_STATES['hard']
            
            #vectors are forced to shrubs when sprouts are hard  
            self.current_p_ts = 1.0  # MUST leave trees
            self.current_p_st = 0.0  # CANNOT move to trees
#-------------More Helper functions---------------------------
    def _get_tree_center(self, x, y):
        if self.cell_type[y, x] != CELL_TYPES['tree']:
            return None
        
        for tree_pos in self.tree_positions:
            cy = tree_pos[0]
            cx = tree_pos[1]
            if abs(cy - y) <= 1 and abs(cx - x) <= 1:
                return (cy, cx)
        return None
#-------------We model the natural aversion to already sick trees as a force------------
    def _repulsion_from_symptomatic_tree(self, vector_index):
        x = self.vector_x[vector_index]
        y = self.vector_y[vector_index]
        
        #find and move to nearby shrub cells from a sick one, first we need to check the status of each surrounding cell
        for dx in [-1, 0, 1]: #8 in neighbourhood
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue #skip current
                new_x = x + dx
                new_y = y + dy
                is_in_bounds = (0 <= new_x < self.width and 0 <= new_y < self.height)
                if is_in_bounds:
                    is_shrub = (self.cell_type[new_y, new_x] == CELL_TYPES['shrub']) #check boundaries
                    if is_shrub:
                        self.vector_x[vector_index] = new_x #pushing vector
                        self.vector_y[vector_index] = new_y
                        return
#-----------------Vector physical dynamics ----------------------
    def vector_movement(self): #this was hard
        vector_life_status_bool = self.vector_alive #begin alive
        alive_indices = np.where(vector_life_status_bool)[0]
        
        for i in alive_indices:
            x = self.vector_x[i]
            y = self.vector_y[i]
            current_cell = self.cell_type[y, x]
            
            #symptomatic trees repel vectors (defined in the function below)
            if current_cell == CELL_TYPES['tree']:
                tree_center = self._get_tree_center(x, y)
                if tree_center:
                    tree_state = self.tree_health.get(tree_center)
                    if tree_state == TREE_STATES['removed']:
                        self._repulsion_from_symptomatic_tree(i)
                        continue
            
            #random movement decision, brownain motion ish
            if random.random() < 0.5:  #attempt movement
                #random direction
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                direction = random.choice(directions)
                dx = direction[0]
                dy = direction[1]
                new_x = x + dx
                new_y = y + dy
                
                #check boundaries
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    new_cell = self.cell_type[new_y, new_x]
                    
                    #check if new position is symptomatic tree
                    if new_cell == CELL_TYPES['tree']:
                        tree_center = self._get_tree_center(new_x, new_y)
                        if tree_center:
                            tree_state = self.tree_health.get(tree_center)
                            if tree_state == TREE_STATES['removed']:
                                continue  #don't move to symptomatic trees
                    
                    move = False #start with no movement
                    if current_cell == CELL_TYPES['shrub']: #i kept gettign crashes from empty lists
                        if new_cell == CELL_TYPES['shrub']:
                            if random.random() < self.p_ss: #defined in main function, maybe would make more sense here
                                move = True
                        elif new_cell == CELL_TYPES['tree']:
                            if random.random() < self.current_p_st:
                                move = True
                    elif current_cell == CELL_TYPES['tree']:
                        if new_cell == CELL_TYPES['tree']:
                            #check if same tree or different
                            if abs(new_x - x) <= 1 and abs(new_y - y) <= 1:
                                if random.random() < self.p_tt:
                                    move = True
                            else:
                                if random.random() < self.p_bb:
                                    move = True
                        elif new_cell == CELL_TYPES['shrub']:
                            if random.random() < self.current_p_ts:
                                move = True
                    
                    if move:
                        self.vector_x[i] = new_x
                        self.vector_y[i] = new_y

    def transmission(self):
        self.daily_new_infections = 0 #for tracking
        vector_life_status_bool = self.vector_alive
        alive_indices = np.where(vector_life_status_bool)[0]
        
        for i in alive_indices:
            x = self.vector_x[i]
            y = self.vector_y[i]
            
            #only transmission on trees
            if self.cell_type[y, x] != CELL_TYPES['tree']:
                continue
            
            tree_center = self._get_tree_center(x, y)
            if tree_center is None:
                continue #if removed
            
            tree_state = self.tree_health[tree_center]
            
            #skip symptomatic trees
            if tree_state == TREE_STATES['removed']:
                continue
            
            #Infected vector to Susceptible tree
            vector_is_infected = self.vector_infected[i]
            tree_is_susceptible = (tree_state == TREE_STATES['susceptible'])
            
            if vector_is_infected and tree_is_susceptible:
                #stochastic transmission with tree susceptibility
                if random.random() < self.tree_susceptibility: #set prior
                    self.tree_health[tree_center] = TREE_STATES['infected']
                    self.infection_time[tree_center] = self.day
                    
                    #symptom appearance time (stochastic also)
                    symptom_delay = np.random.normal(self.symptom_delay_mean, self.symptom_delay_std)
                    symptom_delay = max(0, symptom_delay)
                    self.symptom_time[tree_center] = self.day + symptom_delay
                    self.daily_new_infections += 1
            
            #susceptible vector infected by infected tree
            vector_is_susceptible = not self.vector_infected[i]
            tree_is_infected = (tree_state == TREE_STATES['infected'])
            
            if vector_is_susceptible and tree_is_infected:
                infection_start = self.infection_time.get(tree_center, 0)
                days_infected = self.day - infection_start
                
                #calculate infection level (trees become more infectious)
                if days_infected < 90:  # 3 months of infection
                    colonization_progress = days_infected * self.xylem_propagation_rate
                    colonization = colonization_progress / self.twig_length
                    colonization = min(1.0, colonization)
                    acquisition_prob = self.vector_susceptibility * colonization
                else:
                    acquisition_prob = self.vector_susceptibility
                
                #stochastic acquisition again
                if random.random() < acquisition_prob:
                    self.vector_infected[i] = True

    def update_disease_state(self):
        tree_centers = list(self.tree_health.keys())
        for tree_center in tree_centers:
            current_state = self.tree_health[tree_center]
            if current_state == TREE_STATES['infected']:
                has_symptom_time = tree_center in self.symptom_time
                if has_symptom_time:
                    symptom_day = self.symptom_time[tree_center]
                    if self.day >= symptom_day:
                        self.tree_health[tree_center] = TREE_STATES['removed']
                        #symptomatic trees no longer participate after given time, removed/died#

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

    def _annual_vector_reset(self):
        print(f"Day {self.day}: Annual vector reset")
        
        #reset ALL vectors to susceptible 
        self.vector_alive.fill(True)
        self.vector_infected.fill(False)  #all start susceptible
        
        #redistribute on shrubs - no initially infected vectors
        self.initialize_vectors(initial_infected=0)

    def _record_statistics(self):
        #tree statistics
        states = list(self.tree_health.values())
        total_trees = len(states)
        if total_trees > 0:
            s_count = states.count(TREE_STATES['susceptible'])
            i_count = states.count(TREE_STATES['infected'])
            r_count = states.count(TREE_STATES['removed'])
            
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
            alive_vectors = self.vector_infected[vector_life_status_bool]
            num_infected = np.sum(alive_vectors)
            num_alive = len(alive_vectors)
            infected_fraction = num_infected / num_alive
            self.vector_infection_history.append(infected_fraction)
#-----------------Complicated visual ----------------------
    def visualize(self, show_statistics=True):
        sns.set_style("whitegrid")
        
        if show_statistics: #again to stop crashing from empty lists, prolly don tneed this one
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[:, :2])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[1, 2])
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        
        #main orchard visualization
        display_grid = np.zeros((self.height, self.width, 3))
        
        #shrubs
        SHRUB_bool = self.cell_type == CELL_TYPES['shrub'] #boolean 'mask'
        display_grid[SHRUB_bool] = [0.85, 0.95, 0.85] #rgb needed for matplotlib??
        
        #trees
        for tree_center in self.tree_health:
            state = self.tree_health[tree_center]
            cy = tree_center[0] #as defined before 
            cx = tree_center[1]
            y_start = max(0, cy-1) #finding a 3x3 block of the tree
            y_end = min(self.height, cy+2)
            x_start = max(0, cx-1)
            x_end = min(self.width, cx+2) #min and max stop out of bounds errors
            y_slice = slice(y_start, y_end)
            x_slice = slice(x_start, x_end)
            
            if state == TREE_STATES['susceptible']:
                display_grid[y_slice, x_slice] = [0.2, 0.6, 0.2] #colours for different tree states
            elif state == TREE_STATES['infected']:
                display_grid[y_slice, x_slice] = [0.9, 0.9, 0.3]
            else:  # REMOVED
                display_grid[y_slice, x_slice] = [0.8, 0.2, 0.2]
        
        ax1.imshow(display_grid, interpolation='nearest')
        
        #main olive grove chart setup
        num_trees = len(self.tree_health)
        ax1.set_title(f'Orchard (Day {self.day}) - {num_trees} trees left\n',fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        #tree legend with rgb colours
        legend_elements = [
            patches.Patch(color=[0.2, 0.6, 0.2], label='Susceptible'),
            patches.Patch(color=[0.9, 0.9, 0.3], label='Infected'),
            patches.Patch(color=[0.8, 0.2, 0.2], label='Symptomatic'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        if show_statistics and self.infection_history: #check for none empy list.
            #SIR with seasonal patterns
            days = []
            S = []
            I = []
            R = []
            for record in self.infection_history:
                days.append(record['day'])
                S.append(record['S'])
                I.append(record['I'])
                R.append(record['R'])
            
            ax2.plot(days, S, 'g-', label='Susceptible', linewidth=2)
            ax2.plot(days, I, 'y-', label='Infected', linewidth=2)
            ax2.plot(days, R, 'r-', label='Removed', linewidth=2) #removed rather than recovered as per paper, and obvs as the tree dies
            
            #mark growth phase transitions with backgorund shading
            max_day = max(days)
            num_years = int(max_day/365) + 1
            for year in range(num_years):
                tender_start = year*365 + 90
                tender_end = year*365 + 185
                hard_mid_start = year*365 + 185
                hard_mid_end = year*365 + 365
                hard_early_start = year*365
                hard_early_end = year*365 + 90
                
                label_tender = 'Tender' if year==0 else ''
                label_hard = 'Hard' if year==0 else ''
                
                ax2.axvspan(tender_start, tender_end, alpha=0.1, color='green', label=label_tender)
                ax2.axvspan(hard_mid_start, hard_mid_end, alpha=0.1, color='brown', label=label_hard)
                ax2.axvspan(hard_early_start, hard_early_end, alpha=0.1, color='brown')
            
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Fraction of trees')
            ax2.set_title('SIR with Seasonal Phases', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            #vector infection graph, final graph bottom right
            if self.vector_infection_history:
                ax3.plot(self.vector_infection_history, 'r-', linewidth=2)
                x_vals = range(len(self.vector_infection_history))
                ax3.fill_between(x_vals, self.vector_infection_history, alpha=0.3, color='red')
                ax3.set_xlabel('Days')
                ax3.set_ylabel('Fraction infected')
                ax3.set_title('Vector Infection Rate', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

#-------------running the sim and assigning all the needed values-----------------
def run_simulation(days=365*3, tree_spacing=5, sampling_rate=0.05, time_steps_per_day=48):
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
                      f"R={current['R']:.1%} ")
    
    #statistics
    total_time = time.time() - start_time
    if orchard.infection_history:
        final = orchard.infection_history[-1]
        print(f"S={final['S']:.1%}, I={final['I']:.1%}, R={final['R']:.1%}")
    
    #fig- i should make it an animation
    fig = orchard.visualize(show_statistics=True)
    plt.show()
    
    return orchard


if __name__ == "__main__":
    # Run for 3 years as in paper
    orchard = run_simulation(
        days=365*3, 
        tree_spacing=5, #this was altered in the paper to find optimal spacing
        sampling_rate=0.001,  # 1% sampling
        time_steps_per_day=30  #Paper uses 48 30min steps
    )
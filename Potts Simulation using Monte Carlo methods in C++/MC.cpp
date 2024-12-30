#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib> // to use rand() and srand()
#include <ctime>   // for time()
using namespace std;

const int L = 24;           
const int beta_steps = 50;       // Number of beta values
const int total_steps = 10000; 
const int thermalization = 5000; // steps until thermalization
const int dx[4] = {-1, 1, 0, 0}; //Array to carry the changes in nearest neighbour positions, this is make the function DeltaS more concise
const int dy[4] = {0, 0, -1, 1};
const int q = 3; //q vlaues, number of possible spin values
int grid[L][L];  


//Set our grid to random numbers using the function rand() from a library.
void initialize_grid() {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = rand() % q + 1; // Random spin between 1 and q
        }
    }
}
//Our grid must mimic an infinite lattice so we must simulate periodic boundary conditions
//Function created to impose periodic bc on any grod position
int periodic_bc(int pos) {
    if (pos >= L) {         //if our position falls off the edge of the lattice it appears on the oppoiste side
        return pos - L;
    }
    if (pos < 0){  
        return pos + L;
    }
    return pos;
}

//Our Function to calculate the change in action of our system.
double DeltaS(int x, int y, int new_spin) {
    int old_spin = grid[x][y]; //creating an array of old spins

    if (old_spin == new_spin) {  //Our new spin will be found in the metropolis function
        return 0; // No change if same spin
    }
    //We will compare the action of our new system (after changing a spin) vs. the total action of before
    int old_S = 0;
    int new_S = 0;

    for (int i = 0; i < 4; i++) {
        int nx = periodic_bc(x + dx[i]); //This uses the arrays at the start of the code, otherwise we would have needed many if and else if functions
        int ny = periodic_bc(y + dy[i]); //instead we can cycle through the nearest neighbour positions like so
        int neighbor_spin = grid[nx][ny];

        //We can compare new nearest neighbour spins to that of the new and old grid, thus finding the total change in action
        if (neighbor_spin != old_spin){
             old_S++;
        }
        if (neighbor_spin != new_spin){
             new_S++;
        }
    }

    return double(new_S - old_S);   //return the differen in action
}

//Metropolis function will randomly change the spin of a grid point and decide wether or not to use the change based off our statistics
void metropolis(double beta) {
    for (int step = 0; step < L * L; step++) {
        int x = rand() % L; //We pick a random grid point using modulo as in notes
        int y = rand() % L;
        int old_spin = grid[x][y];    //Define our old spin values as the presvious grod values
        int new_spin = rand() % q + 1; // Proposed new random spin

        if (old_spin == new_spin){ //If there is no change in spin we proceed
             continue;
        }

        double delta_s = DeltaS(x, y, new_spin); //we compute our change in action to decide wether or not to proceed with the spin change.
        if (delta_s <= 0 || (rand() % 10000 / 10000.0) < exp(-beta * delta_s)) {    //using a large number in our rand() we can create a large resolution on the interval [0,1]
            grid[x][y] = new_spin;                                                  //If our spin change increases action (delta S>0) we decide wether to proceed with probability of the boltzmann distrib
        }
    }
}

//Function to calculate our observable Magnetisiation
double calculate_magnetization() {

    int spin_counts[q] = {0};               //We are counting different spins so we need to create an array to hold the count of each spin type

    // we count occurrences of each spin
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            spin_counts[grid[i][j]-1]++;
        }
    }

    // Find the max
    int max_count = 0;
    for (int i = 0; i < q; i++) {
        if (spin_counts[i] > max_count) {
            max_count = spin_counts[i];
        }
    }

    return (double(q) * double(max_count) / double(L * L) - 1.0) / (q - 1.0);    //Retunr the fractional Magnetisation as in notes
}

//Creating a function to run our simulation
void run_montecarlo() {


    initialize_grid(); // Initialize the grid to random varibles

    //Creat eour file in which to store our csv data (as I grpahed using excel)
    ofstream file("mc_results.csv");
    file << "Beta, M, M^2, Variance\n"; //Headers

    //Our beta is our varible so we need to deicde a step size, we are given a resulution in a the prompt
    double step_size = (1.5 - 0.5) / double(beta_steps);
    double total_mag,total_mag_sq; //I defined the varibles outside the forloop to speed up the algorithm
    for (int i = 0; i < beta_steps; i++) {
        double beta = 0.5 + i * step_size; //Vary our beta
        total_mag = 0.0;                    //We need to sum up observables for each beta value
        total_mag_sq = 0.0;

        for (int step = 0; step < total_steps; step++) {
            metropolis(beta);

            // After thermalization, measure magnetization
            if (step >= thermalization) {
                double m = calculate_magnetization();
                total_mag += m;
                total_mag_sq += m * m;
            }
        }

        int steps_after_therm = total_steps - thermalization;   //We only want to consider the steps we took after thermalisation in our calculations
        double mag_avg = total_mag / steps_after_therm;         //FOr each value of beta we want to know the average magnetisation as we used our metropolis function
        double mag_sq_avg = total_mag_sq / steps_after_therm;
        double variance = mag_sq_avg - mag_avg * mag_avg;

        file << beta << "," << mag_avg << "," << mag_sq_avg << "," << variance << "\n";    //storing needed varibales
    }

    file.close();
}

int main() {
    srand(time(0));         //this uses the time function to 'seed' our random number generator, this ensure different results each time we run our function
    run_montecarlo();
    cout << "Results saved to mc_results.csv" << endl;
    return 0;
}

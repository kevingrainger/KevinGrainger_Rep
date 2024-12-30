#include <iostream>
#include <cstring>
#include <cmath>
using namespace std;

const int np = 4;       //Number of planets
double G = 1;           //Gravity const. set to unity         
double t_step = 0.01;  //Time step
double t_final = 5.0;   //End time      


//We neeed to work out the acceleration caused by the the gravitational forces of each planet on each other planet
//I am using 2D arrays, this means was can store a x & y coordinate for each planet
void acc_func(double pos[np][2], double acc[np][2], double mass[np]) {

    //Given that every new position of the system creates a different force
    //We must clear our acceleration each time
    //This is done with a for loop
    for(int i=0; i<np; i++){
        acc[i][0]=0.0;
        acc[i][1]=0.0;
    }

    //We use nested for loops, this means we calculate the interaction between every pair of planets
    //While it prevents double counting
    //[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]    (list of all calculated planet pairs, noting j=i+1)
    for (int i = 0; i < np; ++i) {

        for (int j = i + 1; j < np; ++j) {

            double dx = pos[j][0] - pos[i][0];                      //The distance between planets in cartaesian co-ordinates
            double dy = pos[j][1] - pos[i][1];                      //Note of pos[][0] refers to the x position and pos[][1] to the y position

            double r = sqrt(dx * dx + dy * dy);                     //Radius
            double force_mag = (G * mass[i] * mass[j]) / (r * r);   //Magnitude of gravity force, as given

            double ax_i = (force_mag * dx) / (mass[i] * r);         //Acceleration(s) of planet i
            double ay_i = (force_mag * dy) / (mass[i] * r);         //As derived, when we vectorize of force into x-y directions we need to add factor (dx/r)
            acc[i][0] += ax_i;                                      //These accelerations are added to the 'total acceleration' in each direction
            acc[i][1] += ay_i;                                      //These accelerations only represent one interaction between a pair of planets, not the total acceleration for this location

            double ax_j = (force_mag * dx) / (mass[j] * r);         //Acceleration(s) of planet j
            double ay_j= (force_mag * dy) / (mass[j] * r);
            acc[j][0] -=ax_j;                                       //Forces are in opposite directions (towards each other)
            acc[j][1] -=ay_j;
        }
    }
}

//I have grouped the processes of updating velocity and position into functions for readability in the main ()
//We udate the velocity using our derived formulas
//depending on what form of the leap frog method you wish to use, you can alter the timestep using the function call (eg. update_vel(dt/2))
//- rather than hard coding the factor into the formulae
void update_vel(double vel[np][2], double acc[np][2], double dt) {
    for (int i = 0; i < np; ++i) {

        vel[i][0] += dt * (acc[i][0]);  //x- velocity
        vel[i][1] += dt * (acc[i][1]);  //y- velocity
    }
}

//Position update
void update_pos(double pos[np][2], double vel[np][2], double acc[np][2], double dt) {
    for (int i = 0; i < np; ++i) {

        pos[i][0] += dt * vel[i][0];
        pos[i][1] += dt * vel[i][1];
    }
}

//We need a printing function as we have to cycle through each array element to print it
void Printing_func(double pos[np][2], double time) {

    cout << time;
    for (int i = 0; i < np; ++i) {                      //Printing in csv format , as I used excel to graph the data.
        cout << "," << pos[i][0];}
    for (int i = 0; i < np; ++i) {
        cout << "," << pos[i][1];}
    cout << endl;

}

//Main leapfrog function
//It only calls on all the previous fucntions
void leapfrog(double pos[np][2], double vel[np][2], double acc[np][2], double mass[np], double dt, double t_final) {
    
    double time = 0.0;                            //Initial time value
    cout << "Time,Planet0_x,Planet1_x,Planet2_x,Planet3_x,Planet0_y,Planet1_y,Planet2_y,Planet3_y" << endl;   //Printing data headers in csv format (x-positions to the left and y-positions to the right) 
    

    while (time <= t_final) {
        update_pos(pos, vel, acc, dt/2.0);        //(1) We update position by a half step
        
        acc_func(pos, acc, mass);                 //(2) With this position we can calculate the force and accleration each planet experiences at that point (using acc_func) 

        update_vel(vel, acc, dt);                 //(3) Using the updated acceleration we can calculate velocity

        update_pos(pos, vel, acc, dt/2.0);        //(4)Using velocity we can update our position by a half step once more [we have now advanced a full time step]
        
        
         
        time += dt;         
    }
    Printing_func(pos, time);                 //Next time step 
                   //Printing function can be moved inside the while loop to print all position data
}


int main() {

    double mass[np] = {2.2, 0.8, 0.9, 0.4};                                                 //Filling mass array with initail data
    double pos[np][2] = { {-0.50, 0.10}, {-0.60, -0.20}, {0.50, 0.10}, {0.50, 0.40}};       //Filling position array with initial data   
    double vel[np][2] = {{-0.84, 0.65}, {1.86, 0.70}, {-0.44, -1.50}, {1.15, -1.60}};       //Filling velocity array with initial data
    double acc[np][2];
    leapfrog(pos, vel, acc, mass, t_step, t_final);                                         //Calling our leapfrog function which then calls all other functions

    return 0;
}

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
using namespace std;

const int N = 253;              // Grid dimension
const double tol = 1e-7;       // Tolerance to define convergence
const double omega = 1.8;      // Relaxation factor
double phi[N][N];              // Storing PHI as a 2D array [x][y]
bool boundary[N][N];           // To keep the boundary points constant we need to omit them from the SOR process
//Used a 2D array to store define our boundary points, [x][y]==true if boundary, [x][y]==false if not boundary, we will have a N x N true/false grid.

//Defining the boundaries given for the shapes (boundaries) within the PDE domain//
//Box contour
int box_x1 = 2 * (N-1) / 10;
int box_x2 = 4 * (N-1) / 10;
int box_y1 = 7 * (N-1) / 10;
int box_y2 = 9 * (N-1) / 10;
//Line contour
int line_x = 8 * (N-1) / 10;
int line_y1 = 1 * (N-1) / 10;
int line_y2 = 6 * (N-1) / 10;

//We need to cycle through our grid, assigning the boundary condition values and assigning boundary status boundary[x][y]==true
void boundary_conditions() { 
    memset(boundary, 0, sizeof(boundary));  // Set all elements of the 2D array to false
    //Domain 1x1 square
    for (int i = 0; i < N; i++) {
        double x = double (i) / double (N - 1);
        double y = double (i) / double (N - 1);

        phi[i][N-1] = x;      // Top side= x
        boundary[i][N-1] = true;
        phi[N-1][i] = y;      // Right side= y
        boundary[N-1][i] = true;
        phi[i][0] = 0.0;      // Bottom side= 0
        boundary[i][0] = true;
        phi[0][i] = 0.0;      // Left side= 0
        boundary[0][i] = true;
    }
    //Box x-contour value
    for (int j = box_y1; j <= box_y2; j++) {
        phi[box_x1][j] = 1.0; // Left side of box A
        boundary[box_x1][j] = true;

        phi[box_x2][j] = 1.0;   // Right side of box A
        boundary[box_x2][j] = true;
    }
    //Box y-contour value
    for (int i = box_x1; i <= box_x2; i++) {
        phi[i][box_y1] = 1.0; // Bottom side of box A
        boundary[i][box_y1] = true;

        phi[i][box_y2] = 1.0;   // Top side of box A
        boundary[i][box_y2] = true;
    }
    // Line B contour value
    for (int j = line_y1; j <= line_y2; j++) {
        phi[line_x][j] = 0.0; // Line B with phi = 0
        boundary[line_x][j] = true;
    }
} //End of boundary conditions function

// SOR function to apply the Sucessive Over-Relaxtion method [while keeping boundary conditions fixed]
void SOR() {
    int iterations = 0;
    double global_max_delta = 1.0; // must start at a higher value than tolerance in order to start while loop
    ofstream csvFile("global_max_delta_Dirichlet.csv");
    csvFile << "Iteration,Global_Max_Delta" << endl;
    //while-loop set to end when differecne between current and previous value is within tolerance
    while (global_max_delta >= tol) {
        global_max_delta = 0.0;                       //set back to zero as we only want the maximum delta found for each run

        for (int i = 1; i < N - 1; i++) {      //nested for-loops to cycle through the 2D array [i][j]
            for (int j = 1; j < N - 1; j++) {
                if (boundary[i][j] == false) { // for-loop only runs if the point is not a boundary point

                //new phi value using the SOR increment, as outlined in document
                double phi_1 = (1 - omega) * phi[i][j] + omega * 0.25 * (phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1]);
                // Measure the difference (delta) between new and old phi values
                double delta = fabs(phi_1 - phi[i][j]);

                if (delta > global_max_delta) {      //if our difference is greater than the difference found previously, we assign it as the new delta
                    global_max_delta = delta;       //we are effectivily finding the maximum change in our grid each iteration, and when this is acceptibly small we can say we have convergence
                }
                phi[i][j] = phi_1;
            }  }
        }
       
        if (iterations % 50 == 0){                 //Printing our global max delta value every 50 iterations to monitor convergence.
        csvFile<<iterations<<","<<global_max_delta<<endl;
        }
        iterations++; //increment iteration
    }
    csvFile.close();

    ////////Funciton to print the x,y,phi values if you wish////////
    //Printing the x, y, phi values after convergence
    //I only added this to help track the convergence as the program is running 
    cout << "x, y, phi values after convergence:" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            //we need to transform our grid reference back to x-y values
            double x = (double)i / (N - 1);
            double y = (double)j / (N - 1);
            //cout << "x = " << x << ", y = " << y << ", phi = " << phi[i][j] << endl;
        }
    }
    cout << "Number of iterations " << iterations << endl;      //Printing the number of iterations it took to converge

}   //End of SOR function



double derivative_value() {             // compute the derivative ∂ϕ/∂y(2/5, 1/2)

    //transofrming the x-y coordintes to grid points
    int i = 2 * (N - 1) / 5;           // x = 2/5
    int j = (N - 1) / 2;               // y = 1/2
    double dy = 1.0 / (N - 1);         //Step size dy=dx=(N-1)^-1

     
    return (phi[i][j+1] - phi[i][j-1]) / (2 * dy);  //using finite difference/ first principals
}



int main() {
   
    memset(phi, 0, sizeof(phi));                //setting the phi[x][y] array to zero
    boundary_conditions();                      //setting the initial boundary values
    SOR();                                      //calling our Sucess Over-Relaxation function
    cout << "∂ϕ/∂y(2/5,1/2) =" << derivative_value() << endl;   //Printing the derivstive value for our point

    // printing grid values in CSV format as I used excel to make my graphs
    ofstream csvFile("phi_values.csv");         //Cretaing the CSV file
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            csvFile <<phi[i][j]<<",";          //Storing the Phi values
        }
        csvFile<<endl;                         //Due to excel needing separate columns for 3D graphing, we create a new line each loop
    }
    csvFile.close();

    cout << "phi values in 'phi_values.csv' and global delta values in 'global_max_delta_Dirichlet'" << endl;

    return 0;
}

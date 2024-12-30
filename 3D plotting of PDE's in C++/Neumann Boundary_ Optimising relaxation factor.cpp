#include <iostream>
#include <cmath>
#include <fstream> // For file output
#include <cstring>
using namespace std;

//Our code for the Neumann boundaries conditions does not differ much in set up
//I chose to just add a fucntion to calculate the Neumann b.c's and left the rest as is, even though some parts are redundant.
const int N = 253;              
const double tol = 1e-5;       // tolerance for convergence
const double omega = 1.8;      // Relaxation factor
double phi[N][N];              // 2D array for phi
bool boundary[N][N];           // 2D array to classify boundary points

//Boundary coordinates
// Box contour
int box_x1 = (N-1) / 5;
int box_x2 = 4 * (N-1) / 10;
int box_y1 = 7 * (N-1) / 10;
int box_y2 = 9 * (N-1) / 10;
// Line contour
int line_x = 8 * (N-1) / 10;
int line_y1 = 1 * (N-1) / 10;
int line_y2 = 6 * (N-1) / 10;


void boundary_conditions() {                    //Applies boundary as in Q1 (Dirichlet) to only the top and right sides, aswell as the box and line
    
    memset(boundary, 0, sizeof(boundary));      // Setting all elements of the 2D array boundary[x][y] to false 

    //  Apply Dirichlet conditions on the top and right boundaries
    for (int i = 0; i < N; i++) {
        double x = double(i) / double(N - 1);
        double y = double(i) / double(N - 1);

        phi[i][N-1] = x;      // Top side = x
        boundary[i][N-1] = true;

        phi[N-1][i] = y;      // Right side = y
        boundary[N-1][i] = true;
    }

    // Box x-contour value
    for (int j = box_y1; j <= box_y2; j++) {
        phi[box_x1][j] = 1.0; // Left side of Box A
        boundary[box_x1][j] = true;

        phi[box_x2][j] = 1.0;   // Right side of Box A
        boundary[box_x2][j] = true;
    }

    // Box y-contour value
    for (int i = box_x1; i <= box_x2; i++) {
        phi[i][box_y1] = 1.0; // Bottom side of Box A
        boundary[i][box_y1] = true;

        phi[i][box_y2] = 1.0;   // Top side of Box A
        boundary[i][box_y2] = true;
    }

    // Line B contour value
    for (int j = line_y1; j <= line_y2; j++) {
        phi[line_x][j] = 0.0; // Line B with phi = 0
        boundary[line_x][j] = true;

    }
    
}//end of boundary funcition//

// applying Neumann boundary condition on the left and bottom sides
//No need to check the global_max_delta in this funcition as convergence of the interior points will lead to convergence of the boundaries 
void Neumann_bc() {             

    // Left boundary (∂ϕ/∂x = 0 for x = 0)
    for (int j = 0; j < N; j++) {
        phi[0][j] = (4.0*phi[1][j] - phi[2][j])/3.0; //As derived, we caqn calculate the boundary point using the points 1&2 'steps' into the square

    }
    // Bottom boundary (∂ϕ/∂y = 0 for y = 0)
    for (int i = 0; i < N; i++) {

        phi[i][0] = (4.0*phi[i][1] - phi[i][2])/3.0;
        
    }  
}//End of Neumann b.c function//


// Function to apply the SOR method while keeping boundary conditions fixed
int SOR(double omega) {

    int iterations = 0;
    double global_max_delta = 1.0;         // Initialize to a value larger than tolerance to activate while loop

    while (global_max_delta >= tol) {
        int iterationa=0;
        global_max_delta = 0.0;           //setting max delta back to zero after each run

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                if (boundary[i][j] == false) { // Skip boundary points

                // Calculate the new phi value using the SOR update
                double phi_1 = (1 - omega) * phi[i][j] + omega * 0.25 * (phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1]);

                // Measure the delta between new and old phi values
                double delta = fabs(phi_1 - phi[i][j]);
            
                if (delta > global_max_delta) {         //again finding the max error
                    global_max_delta = delta;
                }

                phi[i][j] = phi_1;
                
                
            }}
        }
        Neumann_bc();                       //N.B we need to call our Neumann function after each iteration of SOR to keep our boundary conditions enforced.
        iterations++;                       // This is because our boundaries depend on the repeatedly changing inner values, so recalculation after SOR is needed
    }

    cout << "Number of iterations: " << iterations << endl;
    return iterations;
}

// derivative ∂ϕ/∂y(2/5, 1/2)
double derivative_value() {
    int i = 2 * (N - 1) / 5;    // x = 2/5
    int j = (N - 1) / 2;        // y = 1/2
    double dy = 1.0 / (N - 1);
    // central difference method
    return (phi[i][j+1] - phi[i][j-1]) / (2 * dy);
}


int main() {
    memset(phi, 0, sizeof(phi));            //Initalizing to 0
    boundary_conditions();                  // Set initial boundary values
    // CSV file to store omega and iterations
    ofstream csvFile("omega_iterations_NBC.csv");
    csvFile << "omega,iterations\n";     //only storing omega vlaues and iterations for graphing, to find optimal omega

    // 'for' a range of omega vales, we call SOR()
    for (double omega = 1.975; omega <= 1.9999; omega += 0.0001) {

        //Important to reset phi and boundary conditions for each omega value (needed as we are not within the SOR funtions)
        memset(phi, 0, sizeof(phi));
        boundary_conditions(); 
        csvFile << omega << "," << SOR(omega) << "\n";  // Call SOR method with given omega, (cycled through by for-loop) & printing the omega and iterations to CSV file
    }

    csvFile.close();
    cout << "Omega-iterations values stored in 'omega_iterations_NBC.csv'" << endl;

    return 0;
}

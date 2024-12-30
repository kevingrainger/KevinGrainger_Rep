#include <iostream>
#include <cmath>
#include <fstream> // For file output
#include <cstring>
using namespace std;

//Unlike previously we will be using less than optimal omega values, thus we need to take measures to shorten run time
const int N = 253;              // A smaller grid size for faster testing of more extreme omega values
const double tol = 1e-5;       // A larger tolerance can shorten run time
double phi[N][N];              // 2D array for phi as before
bool boundary[N][N];           // 2D array to classify boundary points as before

//Defining the boundary co-ordinates
// Box contour
int box_x1 = (N-1) / 5;
int box_x2 = 4 * (N-1) / 10;
int box_y1 = 7 * (N-1) / 10;
int box_y2 = 9 * (N-1) / 10;
// Line contour
int line_x = 8 * (N-1) / 10;
int line_y1 = 1 * (N-1) / 10;
int line_y2 = 6 * (N-1) / 10;

//Adding values to our boundaries as before
void boundary_conditions() { 
    
    memset(boundary, 0, sizeof(boundary));  // Set all elements of the 2D array to false

    // Domain 1x1 square
    for (int i = 0; i < N; i++) {
        double x = double(i) / double(N - 1);   //Converting into grid points
        double y = double(i) / double(N - 1);

        phi[i][N-1] = x;      // Top side = x
        boundary[i][N-1] = true;

        phi[N-1][i] = y;      // Right side = y
        boundary[N-1][i] = true;

        phi[i][0] = 0.0;      // Bottom side = 0
        boundary[i][0] = true;

        phi[0][i] = 0.0;      // Left side = 0
        boundary[0][i] = true;
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
}

// Apply the SOR method while keeping boundary conditions fixed
// returns the number of iterations needed for convergence, so the funciton is no longer void
int SOR(double omega) {
    int iterations = 0;
    double global_max_delta = 1.0; 

    while (global_max_delta >= tol) {
        global_max_delta = 0.0;             //Setting back to zero after each run

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                if (boundary[i][j] == false) { 

                //Calculate the new phi value using the SOR update
                //We will change the omega outside the funciton
                double phi_1 = (1 - omega) * phi[i][j] + omega * 0.25 * (phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1]);

              
                double delta = fabs(phi_1 - phi[i][j]);
                if (delta > global_max_delta) {

                    global_max_delta = delta;
                }

                phi[i][j] = phi_1;
            }}
        }

        iterations++;
    }

    return iterations; // This time we return the number of iterations, makes calling easier in the driver code.
}   //End of SOR function

int main() {

    memset(phi, 0, sizeof(phi));
    boundary_conditions();                  // Set initial boundary values

    // CSV file to store omega and iterations
    ofstream csvFile("omega_iterations.csv");
    csvFile << "omega,iterations\n";     //only storing omega vlaues and iterations for graphing, to find optimal omega

    // 'for' a range of omega vales, we call SOR()
    for (double omega = 1.98; omega <= 1.998; omega += 0.001) {

        //Important to reset phi and boundary conditions for each omega value (needed as we are not within the SOR funtions)
        memset(phi, 0, sizeof(phi));
        boundary_conditions(); 
        csvFile << omega << "," << SOR(omega) << "\n";  // Call SOR method with given omega, (cycled through by for-loop) & printing the omega and iterations to CSV file
    }

    csvFile.close();
    cout << "Omega-iterations values stored in 'omega_iterations.csv'" << endl;

    return 0;
}

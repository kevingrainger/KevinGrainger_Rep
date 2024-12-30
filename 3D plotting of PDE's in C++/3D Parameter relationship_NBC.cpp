#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
using namespace std;


int N = 353;                   // Grid dimension (will be changed in main)
double tol = 1e-8;             // Tolerance (will be changed in main)
const double omega = 1.8;      
double phi[500][500];          // max size N = 500 as we are going to be changing N
bool boundary[500][500];       // boundary point array


int box_x1 = 2 * (N - 1) / 10;
int box_x2 = 4 * (N - 1) / 10;
int box_y1 = 7 * (N - 1) / 10;
int box_y2 = 9 * (N - 1) / 10;
int line_x = 8 * (N - 1) / 10;
int line_y1 = 1 * (N - 1) / 10;
int line_y2 = 6 * (N - 1) / 10;

// boundary conditions function
void boundary_conditions() { 
    memset(boundary, 0, sizeof(boundary));  // Set all elements of the boundary array to false  
    // Domain 1x1 square
    for (int i = 0; i < N; i++) {
        double x = double(i) / double(N - 1);
        double y = double(i) / double(N - 1);

        phi[i][N - 1] = x;       
        boundary[i][N - 1] = true;
        phi[N - 1][i] = y;       
        boundary[N - 1][i] = true;
        phi[i][0] = 0.0;         
        boundary[i][0] = true;
        phi[0][i] = 0.0;         
        boundary[0][i] = true;
    }
    // Box x-contour
    for (int j = box_y1; j <= box_y2; j++) {
        phi[box_x1][j] = 1.0;    
        boundary[box_x1][j] = true;

        phi[box_x2][j] = 1.0;    
        boundary[box_x2][j] = true;
    }
    // Box y-contour
    for (int i = box_x1; i <= box_x2; i++) {
        phi[i][box_y1] = 1.0;    
        boundary[i][box_y1] = true;

        phi[i][box_y2] = 1.0;    
        boundary[i][box_y2] = true;
    }

    // Line contour
    for (int j = line_y1; j <= line_y2; j++) {
        phi[line_x][j] = 0.0;    
        boundary[line_x][j] = true;
    }
}

// function Neumann boundary condition (∂ϕ/∂x = 0, ∂ϕ/∂y = 0)
void Neumann_bc() { 
    // Left boundary (∂ϕ/∂x = 0 for x = 0)
    for (int j = 0; j < N; j++) {
        phi[0][j] = (4.0 * phi[1][j] - phi[2][j]) / 3.0; 
    }
    // Bottom boundary (∂ϕ/∂y = 0 for y = 0)
    for (int i = 0; i < N; i++) {
        phi[i][0] = (4.0 * phi[i][1] - phi[i][2]) / 3.0;
    }
}

// SOR method
void SOR() {
    int iterations = 0;
    double global_max_delta = 1.0; 

    while (global_max_delta >= tol) {
        global_max_delta = 0.0;   // reset the maximum delta for this iteration

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                if (!boundary[i][j]) { 
                    double phi_1 = (1 - omega) * phi[i][j] + omega * 0.25 * 
                                   (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]);
                    double delta = fabs(phi_1 - phi[i][j]);

                    if (delta > global_max_delta) {
                        global_max_delta = delta; // Update max delta
                    }
                    phi[i][j] = phi_1;
                }
            }
        }

        // Apply the Neumann boundary condition after each SOR iteration
        Neumann_bc(); 
        iterations++;
    }

    //cout << "Number of iterations: " << iterations << endl;
}

// ∂ϕ/∂y(2/5, 1/2) function
double derivative_value() {
    int i = 2 * (N - 1) / 5;     
    int j = (N - 1) / 2;         
    double dy = 1.0 / (N - 1);   

    return (phi[i][j + 1] - phi[i][j - 1]) / (2 * dy); 
}


int main() {
    ofstream csvFile("Parameter_Opt.csv");
    csvFile << "N,tol,derivative_value" << endl;

    // The only change from the other programs is that we must call our function in a loop over different N and tolerance values
    //And apply changes by assigning the N and tol variablies to a temporary varibel that is being changed in the for-loop
    for (int altered_N = 50; altered_N <= 500; altered_N += 50) {//Nested for-loop so we can get every combination of parameters
        for (double altered_tol = 1e-1; altered_tol >= 1e-8; altered_tol /= 10) {
            N = altered_N;      // change grid dimension
            tol = altered_tol;  // change tolerance

            memset(phi, 0, sizeof(phi));  // set phi array to zero
            boundary_conditions();       // set boundary conditions ==true
            SOR();                      
            csvFile << N << "," << altered_tol << "," << derivative_value() << endl; //Note that we print 'N' and 'altered_tol' into the csv file as we are within the second for-loop
        }
    }
    csvFile.close();
    cout<<"N,tol, & derivative values saved to Parameter_Opt.csv"<<endl;
    return 0;
    
}

using namespace std;
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

double t0 = 0.0;            // Starting time
double t1 = 10.0;           // End time
double x_0 = 3.0 / 4.0;     // Given condition x(0) = 3/4
double x_10 = -1.0;         // Given condition x(10) = -1

//derivative_array array stores the derivatives of a X[] array (needed for the Rk4 method)
//We set it to represent our system of ODE's
vector<double> derivative_array(double t, vector<double> X){
    return vector<double> {X[1],-t * X[0] * (t + 2) / (2 + t * t * X[0] * X[0])}; // Defining manually our ODE as two 1st order ODE's in a vector form
}

//The Vector calculation needed for the RK4 method
vector<double> AddMul(vector<double> U, vector<double> V, double a){    
    for(int i=0; i<V.size(); i++){
        V[i] = U[i] + a*V[i];      //We have a for-loop, scaling every value of V[i] and adding it to vector U[i] (Generalized calculation needed in RK4)
    }
    return V;
}


//RK4 returns final value of X0 array at time tstop. Saves intermediate values in csv file to be plotted
vector<double> RK4_Save_Print(vector<double> X0, double t0, double tstop, int n){
    ofstream ofs;                                      //We are storing the results in a csv file as there were many data points
    ofs.open("X0_array.csv");

    double h=(tstop-t0)/(double(n));
    vector<double> k1, k2, k3, k4;                     //Our k-formulae become 2D vectors, as we are solving both ODE's in our system    
    vector<double> X_temp;                             //Needed so as to not overwrite our original X0[] 

    ofs<<"time,"<<"x(t),"<<"x'(t)"<<endl;              //Saves headers to csv 
    cout<<"time,"<<"  x(t),"<<"x'(t),"<<endl;          //Prints headers
    for(int i=1; i<=n; i++){
        X_temp = X0;                                   //sets X_temp to the intial conditions
        k1=derivative_array(t0,X_temp);                //k1 calculated in terms of derivative_array function with imputed initial conditions (note this is a 2-d vector)

        X_temp = AddMul(X0, k1, 0.5*h);                //X_temp is advanced by half step with AddMul function. (X_temp[] is altered, X0[] intact)
        k2=derivative_array(t0+0.5*h,X_temp);          //k2 uses the altered X_temp, ie. a half step added

        X_temp = AddMul(X0, k2, 0.5*h);                //process continues.
        k3=derivative_array(t0+0.5*h,X_temp);

        X_temp = AddMul(X0, k3, 1.0*h);
        k4=derivative_array(t0+h, X_temp);
         
        for (int i = 0; i < X0.size(); ++i) {
            X0[i] = X0[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);    //We need to average the results in the standard Runge Kutta weighting, using for-loop to access every element
        }
        t0+=h;
        ofs<<t0<<","<<X0[0]<<","<<X0[1]<<endl;          //storing our csv data
        cout<<t0<<","<<X0[0]<<","<<X0[1] <<","<<endl;   //printing
    }
	ofs.close();
    return X0;                                          //returning our final X0 array
}


//Shooting Method
double shooting_method(double dxdt_guess, double h) {
    
    vector<double> X0 = {x_0, dxdt_guess};                                           //Defining the 'initial conditions' vector X0 as: x0, and random guess for x'(t). X0=[x0,x't(guess)]
    vector<double> X_RK4 = RK4_Save_Print(X0, t0, t1, int((t1 - t0) / h));           //We run RK4 with this guess for x'(t) as an input via X0, we are running the Rk4 with our guess at the intial conditions
                                                                        
    return X_RK4[0] - x_10;                                                          //We produce a function whose roots correspond to the correct guess of x'(t)
}

//Bisection Method
//Needed to solve the returned equation from the shooting method (X_RK4[0] -x1)
//Saves revised boundary [x'(0)] for bisection method in csv file
double bisect(double x_a, double x_b,double h){                         //Defining our area boundaries, x_a, x_b , in which we search for a root
    
    double f_xa = shooting_method(x_a, h);                              //Finding the result of our RK4 with x_a (root guess) as the input                             
    double f_xb =shooting_method(x_b, h);                               //with x_b as the guess
    double x_c= 0.5*(x_a+x_b);                                          //Finds the middle of these boundaries

    ofstream ofs;
    ofs.open("Guesses.csv");


    while (x_b-x_a >1.0e-6){                        //We set our tolerance for how small our 'section' needs to be

        cout<<"x'(0) estimate "<<x_c<<endl;         //Printing our estimate after each iteration of the bisection method, to see the evolution of x'(t=0)
        ofs<<x_c<<endl;                             //Saving to csv file 

        double f_xc = shooting_method(x_c,h);       //plug our middle value x_c into our shooting method funcition, which will use it in the RK4
        if(f_xa*f_xc < 0.0){                        //If the product of our functions (x_a & x_c) is negative it means we have a root in this section, as the function has crossed the axis
            x_b=x_c;                                //We then reassign the boundaries to make the search area smaller
            f_xb = f_xc;                            //Reassign our functions
        }
        else{                                       //In the case the product is positive we know the root is not located in that area, so we reassign the borders in the opposite direciton
            x_a=x_c;
            f_xa = f_xc;
        }
        x_c=0.5*(x_a+x_b);                          //We then take the mid point of our new section and repeat the process
    }
    ofs.close();
    return 0.5*(x_a+x_b);                           //We return the final point x_c, or in other words the average of x_a and x_b (these are very close in value) 
}

int main(){

    double h=0.01;                                   //step size
    //x'(t=0) -> bisect(-1.6,-1.3,h) our bisection function finds the actual value of x'(t=0), we give a wide range [-3,3]
    double dx_0=bisect(-3.0,3.0,h);
    vector<double> X0 = {x_0,dx_0};                 //Our X0 vector is now the given initial condition x_0 and the calculated x'(t=0) [using bisection method] 
    int n = (t1-t0)/h ;                             //Defining our number of iterations
    RK4_Save_Print(X0,t0,t1,n);                     //We use the now complete initial condition 'X0' vector in the Rk4 method to solve the ODE!
    cout<< "Final x'(0)="<<dx_0<<endl;              //Printing the value of x'(t) 

    return 0;
}
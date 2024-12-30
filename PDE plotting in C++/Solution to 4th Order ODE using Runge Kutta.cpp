using namespace std;
#include <stdio.h>
#include <math.h>
#include <iostream>

//Defining our analytical solution as a function for comparison.
//x(t)=exp(1/3 t^3 -2t^2 +4t) -1
double Asolution(double t){
    return exp((pow(t,3))/3.0 - 2*pow(t,2) + 4*t)-1.0;
}

//Isolating the derivative on the left of the ODE equation means we can define a 'slope' function
//We can define the k_equations in terms of this function
//dx/dt=(t-2)^2(x+1)
double slope(double t, double x){
    return (pow(t-2,2)*(x+1));
}


//defining the Runge Kutta function
//This will produce the final x solution with inputs of the initial conditions (x0,t0) & step size h
double RK4(double t0, double x0, double t, double h){

    //defining the number of steps in terms of the step size
    //We need this variable to tell the for-loop when to stop
    int n=int ((t-t0)/h);
    h=((t-t0)/(double(n)));
    double k1, k2, k3, k4;
    //starting at initial x
    double x=x0;
    
    //printing headings for data
    cout << "t" << " " << "RK4" <<" "<<"Analytic"<<" "<< "Error"<<endl;
    
    //for-loop to iterate the RK4 a given number of time 'n' calculated by inputted 'h'
    for(int i=1; i<=n; i++){
    
        //Calculating and printing of the Analytic solution for calculation of error at every iteration
        double Asol=Asolution(t0);
        //fabs gives us the absolute error value
        cout << t0 << " " << x <<" "<<Asol<<" "<< fabs(x-Asol)<<endl;

        //Runge kutta equations defined in temrs of our 'slope# function
        k1= h*slope(t0,x);
        k2=h*slope(t0+0.5*h,x+0.5*k1);
        k3=h*slope(t0+0.5*h,x+0.5*k2);
        k4=h*slope(t0+h, x+k3);

    //X is iterated with a weighted average of our k_formulae
    x+=(k1+2*k2+2*k3+k4)/(6.0);
    //t is increased by step size h
    t0+=h;
    }
    return x;
}

int main(){
    //setting up our initial conditions to insert into our called RK4 function
    double t0=0.0;
    double x0=0.0;     
    double t=1.0; //final time
    double h=0.1; //step size
    
    RK4(t0,x0,t,h);
    return 0;
}
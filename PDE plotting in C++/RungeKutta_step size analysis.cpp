using namespace std;
#include <stdio.h>
#include <math.h>
#include <iostream>


//Similar form to part (i) of Q1 
double Asolution(double t){
    return exp((pow(t,3.0))/3.0 - 2.0*pow(t,2) + 4.0*t)-1.0;
}

double slope(double t, double x){
    return (pow(t-2.0,2.0)*(x+1));
}



//RK4 function
double RK4(double t0, double x0, double t, int n){

    //Changed from part (i)
    //Defining our step size in terms of 'double n'
    //This means there is no need to round our n value if h does not divide equally into total step
    double h=(t-t0)/(double(n));
    double k1, k2, k3, k4;
    double x=x0;
    
    //no changes to RK4 for-loop
    for(int i=1; i<=n; i++){
    
        
        k1= h*slope(t0,x);
        k2=h*slope(t0+0.5*h,x+0.5*k1);
        k3=h*slope(t0+0.5*h,x+0.5*k2);
        k4=h*slope(t0+h, x+k3);

        x+=(k1+2*k2+2*k3+k4)/(6.0);
        t0+=h;
    
    }
    //We only want to print the final Analytical solution
    double Asol=Asolution(t0);
    return x;
}

//we can define a tolerance to shut off a while-loop once the accuracy is reached
int main(){
    //Defining intial conditions
    double t0=0.0; 
    double t=1.0;
    double n=10;
    double x0=0.0;
    double tol = 1e-8;
    double xstop, error, Asol; 
    //Short-hand for printing
    //We need to initialise these values before they enter the for-loop
    xstop = RK4(t0, x0, t, n); //final x vlaue
    Asol = Asolution(t);    //Analytic solution
    error = fabs(xstop - Asol); //absolute error

    //printing headers
    cout<<"step size"<<" "<<"Steps"<<" "<<"RK4"<<" "<<"Analytic"<<" "<<"error"<<" "<<"h^4"<<" "<< "log(error)"<<" "<< "4log(h)"<<endl;

    //Our while loop running until error is beneath tolerance (accurate to 8th order)
    while (error > tol){
        
        n+=1;   //increase our step number
        xstop=RK4(t0,x0,t,n);   //calculate Rk4 for this number of steps
        Asol=Asolution(t);      //calculate Analytic solution
        error= fabs(xstop-Asol);    //calculate error
    
    //Printing the solutions and error for each number of steps
    std::cout   <<(t-t0)/n<<" "
                << n <<" "
                << xstop <<" "
                << Asol <<" "
                << error <<" "
                << pow((t-t0)/n,4)<<" "
                << log10(error)<<" "
                << 4*log10((t-t0)/n)<<std::endl;
        
    }
    //printing the final data once we have reached required accuracy
    cout<<"\nAnalytical Solution"<< Asol<<endl; 
    cout<<"\nRK4 Solution"<< xstop<<endl; 
    cout<<"\nerror"<< error<<endl; 
    cout<<"\nFinal number of steps"<< n<<endl; 
    cout<<"\nFinal Step Size"<<(t-t0)/n<<endl;  //calculating the step size from the final number of steps

    return 0;
}
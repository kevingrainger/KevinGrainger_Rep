using namespace std;
#include <cmath>
#include <vector>
#include <iostream>

//Once we have our system of lower order ODE's in the form of vectors
//vector of the function 'slope' which holds the derivative values
// Returns Derivative of Y(x)
vector<double> slope(double x, vector<double> Y){
    return vector<double> {Y[1], Y[2], Y[3], (-10.0*Y[2] -1.0/7.0 * x * pow(Y[0],3))};
}

//To use vectors we need to also define a vector operation to use in the RK4 process
//Takes imputted vector 'V', adds it to the vector of intial conditions 'U', then 'V' is scaled by a step size 'a'
vector<double> AddMul(vector<double> U, vector<double> V, double a){
    for(int i=0; i<V.size(); i++){
        V[i] = U[i] + a*V[i];
    }
    return V;
}

//RK4 function
vector<double> RK4(vector<double> Y0, double x0, double xstop, int n){

    double h=(xstop-x0)/(double(n));
    vector<double> k1, k2, k3, k4;  //all formulae must be vectors of 4, we need 16 equation to solve this system
    vector<double> Y=Y0;    //setting our varible to the inital conditions
    vector<double> Y_temp;  //New varible needed to stop the overwritting over our original Y[] as it passes through the k_formulae
    
    for(int i=1; i<=n; i++){
        Y_temp = Y;                         //sets y_temp to the intial conditions
        k1=slope(x0,Y_temp);                //k1 calculated in terms of slope function with imputed initial conditions (note this is a 4-d vector)

        Y_temp = AddMul(Y, k1, 0.5*h);      //Y_temp is advanced by half step with AddMul function. (Y_temp[] is altered, Y[] intact)
        k2=slope(x0+0.5*h,Y_temp);          //k2 uses the altered Y_temp, ie. a half step added

        Y_temp = AddMul(Y, k2, 0.5*h);      //process continues.
        k3=slope(x0+0.5*h,Y_temp);

        Y_temp = AddMul(Y, k3, 1.0*h);
        k4=slope(x0+h, Y_temp);
        //We are using Y_temp to make sure the we input a non-altered value of Y into each k_formula
        //We are left four 4-D vectors (16 values) these correspond to the 4 k_values for each ODE in our system
        //This could have been complete explicitly with 4 sets of 4 equaitions

        //normal weighted average
        //We need to add the corresponding k_equtions to each initial element of Y so a for loop is used 
        for (int i = 0; i < Y.size(); ++i) {
            Y[i] = Y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        x0+=h;
        cout << x0 <<" "<< Y[0] << endl;       //Printing Y[0] allows us to see the evolution of the first element which is y_1, and we assigned y_1 = y
    }
    return Y;
}


int main(){     
    //from intial condt.
    double x0=0.0;
    double xstop=30.0;
    double n=1000;
    double tol = 1e-8;
    vector<double> Y0 = {1,0,0,0};  //setting up our vecotr with inital conditions
    cout << "X values"<< " " <<"Y values" <<endl;  //heading for data
    cout <<RK4(Y0, x0, xstop, n)[0] << endl;   //calling the Rk4 with the starting conditions and calling solution
    return 1;
}

    

 

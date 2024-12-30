#include <iostream>
#include <cmath>
#include <iomanup>
#include <vector>
#include <functional>

// Parameters
const double T0 = 0.0;       // Initial time
const double T1 = 10.0;      // Final time
const double X0 = 3.0 / 4.0; // Initial condition x(0) = 3/4
const double X1 = -1.0;      // Boundary condition x(10) = -1

// Define the system of first-order ODEs
std::vector<double> system_of_odes(double t, const std::vector<double>& y) {
    std::vector<double> dydt(2);
    dydt[0] = y[1]; // dy1/dt = y2 (which is dx/dt)
    dydt[1] = -t * y[0] * (t + 2) / (2 + t * t * y[0] * y[0]); // dy2/dt
    return dydt;
}

// Runge-Kutta 4th order method
std::vector<double> runge_kutta_4(const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
                                  double t, const std::vector<double>& y, double h) {
    std::vector<double> k1 = f(t, y);
    std::vector<double> y_temp(y.size());
    
    for (size_t i = 0; i < y.size(); ++i) y_temp[i] = y[i] + 0.5 * h * k1[i];
    std::vector<double> k2 = f(t + 0.5 * h, y_temp);
    
    for (size_t i = 0; i < y.size(); ++i) y_temp[i] = y[i] + 0.5 * h * k2[i];
    std::vector<double> k3 = f(t + 0.5 * h, y_temp);
    
    for (size_t i = 0; i < y.size(); ++i) y_temp[i] = y[i] + h * k3[i];
    std::vector<double> k4 = f(t + h, y_temp);
    
    std::vector<double> y_next(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y_next[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return y_next;
}

// Function to integrate the system using RK4
std::vector<double> integrate_rk4(double t0, double t1, std::vector<double> y0, double h,
                                  const std::function<std::vector<double>(double, const std::vector<double>&)>& f) {
    double t = t0;
    std::vector<double> y = y0;
    
    while (t < t1) {
        y = runge_kutta_4(f, t, y, h);
        t += h;
    }
    return y;
}

// Shooting method
double shooting_method(double initial_guess, double h) {
    std::vector<double> y0 = { X0, initial_guess }; // y0 = { x(0), x'(0) }
    
    // Integrate from t = 0 to t = 10 using RK4
    std::vector<double> y_end = integrate_rk4(T0, T1, y0, h, system_of_odes);
    
    // Return the difference from the boundary condition at t = 10
    return y_end[0] - X1;
}

// Bisection method to refine the guess for x'(0)
double find_initial_velocity(double h, double tol = 1e-6) {
    double v_low = -10.0, v_high = 10.0;
    double v_mid, f_low, f_mid;

    f_low = shooting_method(v_low, h);
    
    while (v_high - v_low > tol) {
        v_mid = 0.5 * (v_low + v_high);
        f_mid = shooting_method(v_mid, h);
        
        if (f_mid * f_low < 0) {
            v_high = v_mid;
        } else {
            v_low = v_mid;
            f_low = f_mid;
        }
    }
    return 0.5 * (v_low + v_high);
}

int main() {
    double h = 0.01; // Time step
    double initial_velocity = find_initial_velocity(h);
    
    std::cout << "The initial velocity x'(t=0) to 6 significant figures is: " 
              << std::fixed<< initial_velocity << std::endl;
    
    return 0;
}

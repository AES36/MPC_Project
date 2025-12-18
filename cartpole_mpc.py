import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class CartPole:
    def __init__(self, M=1.0, m=0.1, l=0.5, g=9.81):
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.state = np.zeros(4) # [x, x_dot, theta, theta_dot]

    def dynamics(self, state, u):
        # Nonlinear dynamics
        x, x_dot, theta, theta_dot = state
        force = u[0]
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Denominator for both equations
        denom = self.M + self.m * (1 - cos_theta**2)
        
        # Acceleration of theta (angular acceleration)
        theta_acc = (self.g * sin_theta * (self.M + self.m) - 
                     force * cos_theta - 
                     self.m * self.l * theta_dot**2 * sin_theta * cos_theta) / (self.l * denom)
                     
        # Acceleration of x (linear acceleration)
        x_acc = (force + self.m * self.l * (theta_dot**2 * sin_theta - theta_acc * cos_theta)) / (self.M + self.m)
        
        return np.array([x_dot, x_acc, theta_dot, theta_acc])

    def step(self, u, dt):
        # RK4 Integration
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * dt * k1, u)
        k3 = self.dynamics(self.state + 0.5 * dt * k2, u)
        k4 = self.dynamics(self.state + dt * k3, u)
        
        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state

class MPCController:
    def __init__(self, M=1.0, m=0.1, l=0.5, g=9.81, dt=0.1, N=10):
        self.dt = dt
        self.N = N # Horizon
        
        # Linearized Model Matrices (Continuous)
        # Derived around theta=0, theta_dot=0, u=0
        
        # Mass Matrix MM = [[M+m, ml], [ml, ml^2]]
        MM = np.array([[M+m, m*l], [m*l, m*l**2]])
        MM_inv = np.linalg.inv(MM)
        
        # Coefficients for theta and u
        # [x_dd; theta_dd] = MM_inv * [0; mgl] * theta + MM_inv * [1; 0] * u
        coeff_theta = MM_inv @ np.array([0, m*g*l])
        coeff_u = MM_inv @ np.array([1, 0])
        
        # State: [x, x_dot, theta, theta_dot]
        self.Ac = np.zeros((4,4))
        self.Ac[0, 1] = 1
        self.Ac[2, 3] = 1
        self.Ac[1, 2] = coeff_theta[0]
        self.Ac[3, 2] = coeff_theta[1]
        
        self.Bc = np.zeros((4,1))
        self.Bc[1, 0] = coeff_u[0]
        self.Bc[3, 0] = coeff_u[1]
        
        # Discretize (Forward Euler)
        self.Ad = np.eye(4) + self.Ac * dt
        self.Bd = self.Bc * dt
        
        # Weights
        self.Q = np.diag([10.0, 1.0, 100.0, 1.0]) 
        self.R = np.diag([0.1])
        
    def predict(self, x0, U):
        # Predict state trajectory given initial state x0 and input sequence U
        # U is flat array of size N
        x_curr = x0.copy()
        cost = 0
        
        for k in range(self.N):
            u_k = U[k]
            
            # State Cost
            cost += x_curr.T @ self.Q @ x_curr
            # Input Cost
            cost += u_k**2 * self.R[0,0]
            
            # Dynamics Update
            x_curr = self.Ad @ x_curr + self.Bd.flatten() * u_k
            
        # Terminal Cost
        cost += x_curr.T @ (self.Q * 10) @ x_curr
        return cost

    def solve(self, current_state):
        # Initial guess for U (zeros)
        u0 = np.zeros(self.N)
        
        # Bounds for u
        bounds = [(-20.0, 20.0) for _ in range(self.N)]
        
        # Optimization
        res = minimize(lambda U: self.predict(current_state, U), 
                       u0, 
                       bounds=bounds, 
                       method='SLSQP',
                       options={'ftol': 1e-4, 'disp': False})
        
        if res.success:
            return np.array([res.x[0]])
        else:
            # Fallback if optimization fails (though SLSQP is robust)
            return np.array([res.x[0]])

def main():
    # Parameters
    dt = 0.05
    T_max = 10.0
    steps = int(T_max / dt)
    
    # Initialize
    # Initialize
    cartpole = CartPole()

    mpc = MPCController(dt=dt, N=15) # Reduced N slightly for speed with scipy
    
    # Initial State
    theta0_deg = np.random.uniform(-20, 20)
    theta0_rad = np.radians(theta0_deg)
    
    cartpole.state = np.array([0.0, 0.0, theta0_rad, 0.0])
    
    # History
    t_hist = []
    x_hist = []
    theta_hist = []
    u_hist = []
    
    print(f"Starting Simulation with theta0 = {theta0_deg:.2f} degrees")
    
    for i in range(steps):
        # Get Control
        u = mpc.solve(cartpole.state)
        
        # Step Dynamics
        cartpole.step(u, dt)
        
        # Record
        t_hist.append(i * dt)
        x_hist.append(cartpole.state[0])
        theta_hist.append(cartpole.state[2])
        u_hist.append(u[0])
        
        if i % 10 == 0:
            print(f"Time: {i*dt:.2f}s, Theta: {np.degrees(cartpole.state[2]):.2f} deg, x: {cartpole.state[0]:.2f} m")

    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_hist, np.degrees(theta_hist))
    plt.ylabel('Theta (degrees)')
    plt.title(f'Cart-Pole Stabilization (Initial: {theta0_deg:.2f} deg)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t_hist, x_hist)
    plt.ylabel('Position (m)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t_hist, u_hist, 'r')
    plt.ylabel('Control Force (N)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cartpole_mpc_results.png')
    print("Simulation complete. Results saved to cartpole_mpc_results.png")

if __name__ == "__main__":
    main()

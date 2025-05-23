import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravity (m/s^2)

# Metronome Parameters
m_pendulum = 0.1  # Mass of pendulum (kg)
L1_pendulum = 0.3  # Length of first pendulum (m)
L2_pendulum = 0.3  # Length of second pendulum (m)
epsilon = 0.15  # Escapement force coefficient
theta1_0_init=theta1_0 = 0.9  # Escapement threshold angle for metronome 1 (radians)
theta2_0_init=theta2_0 = 0.25  # Escapement threshold angle for metronome 2 (radians)
I1 = m_pendulum * L1_pendulum**2  # Inertia of first pendulum
I2 = m_pendulum * L2_pendulum**2


control_mode =  'PD'       #FLC(Feedback Linearization Control),NC(no control),PD(pd control)

# Control Parameters
Kp_PD = 1.0  # Proportional gain for PD control
Kd_PD = 0.05# Derivative gain for PD control
A_PD = 0.25  # Amplitude of reference sine signal for PD control
omega_PD = 0.5 # Frequency of reference signal
kd1 = kd2 = kd3 = 0.01
kp1 = kp2 = kp3 = 0.8

Kp_FB = np.diag([kp1, kp2, kp3])  # Proportional gain for feedback linearization
Kd_FB = np.diag([kd1, kd2, kd3])  # Derivative gain for feedback linearization

# Surface Parameters (Coupling)
M_surface = 1.0  # Mass of the surface (kg)
B_surface = 0.1  # Surface damping coefficient
K_surface = 0.5  # Surface stiffness


# Initial Conditions (Angles in radians)
omega1_0 = 0.0  # Initial angular velocity
omega2_0 = 0.0  # Initial angular velocity
x_0 = 0.0  # Initial displacement of the surface
v_0 = 0.0  # Initial velocity of the surface

# Time parameters
T_max = 30  # Simulation duration (s)
dt = 0.01  # Time step

# Cart Parameters
M_cart = 1.0  # Mass of the cart (kg)
B_cart = 0.1  # Damping coefficient
K_cart = 0.5  # Stiffness of the cart
#tau_input = 0# input torque

def affiche(T,t):
    # Surface Movement
    plt.subplot(2, 1, 1)
    #plt.xlim(0,15)
    #plt.ylim(-1.5,1.5)
    plt.plot(time, T, label="Surface Displacement", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Surface Displacement (m)")
    plt.title("Surface Vibration Due to Metronomes")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    


def metronome_matrix_system(t, y):
    theta1, omega1, theta2, omega2, x, v = y  # Unpack state variables

    # Mass Matrix M(q)
    M_q = np.array([
        [I1, 0, m_pendulum * L1_pendulum * np.cos(theta1)],
        [0, I2, m_pendulum * L2_pendulum * np.cos(theta2)],
        [m_pendulum * L1_pendulum * np.cos(theta1), m_pendulum * L2_pendulum * np.cos(theta2), M_cart + 2 * m_pendulum]
    ])

    # Coriolis and Damping Matrix C(q, q_dot)
    C_q_qdot = np.array([
        [epsilon * ((theta1 / theta1_0)**2 - 1) * m_pendulum*(L1_pendulum**2), 0, 0],
        [0, epsilon * ((theta2 / theta2_0)**2 - 1) *m_pendulum*(L2_pendulum**2) , 0],
        [m_pendulum * L1_pendulum * np.sin(theta1) * omega1, m_pendulum * L2_pendulum * np.sin(theta2) * omega2, 0]
    ])

    # Gravity Matrix G(q)
    G_q = np.array([
        [m_pendulum * g * L1_pendulum * np.sin(theta1)],
        [m_pendulum * g * L2_pendulum * np.sin(theta2)],
        [0]  # No gravitational force directly on the cart
    ])
    # Control Input τ
    if control_mode == "PD":
        # PD control to track a sinusoidal trajectory
        q_d = A_PD * np.sin(omega_PD * t)  # Reference trajectory
        q_tilde = q_d - x  # Tracking error
        tau_input = Kp_PD * q_tilde - Kd_PD * v  # PD control law



    elif control_mode == "FLC":
        # Feedback Linearization: Track Metronome 1's motion
        q_d = [theta1 ,theta1,-theta1] # Desired trajectory (Metronome 1)
        qq =[omega1,omega1,-omega1]
        q_tilde = q_d - np.array([[theta1], [theta2], [x]])  # Error between cart and metronome
        q_titlde_dot = qq - np.array([[omega1], [omega2], [v]]) 
        v_control = Kp_FB @  q_tilde+ Kd_FB @ q_titlde_dot
        tau_input = (M_q @ v_control + C_q_qdot @ np.array([[omega1], [omega2], [v]]) + G_q)[2, 0]  # Compute τ

    #CONTROL LOW
        # Compute intermediate terms
        K1 = m_pendulum * L1_pendulum * np.cos(theta1) * (kp1*q_tilde[0]+kd1*q_titlde_dot[0])
        K2 = m_pendulum * L2_pendulum * np.cos(theta2) * (kp2*q_tilde[1]+kd2*q_titlde_dot[1])
        K3 = (M_cart + 2 * m_pendulum)  * (kp3*q_tilde[1]+kd3*q_titlde_dot[2])

        # Compute control input τ
        tau = K1 + K2 + K3 - (m_pendulum * L1_pendulum * np.sin(theta1) * omega1**2) - (m_pendulum * L2_pendulum * np.sin(theta2) * omega2**2)
   
    

    elif control_mode=="NC":
        tau_input = 0  # No control applied
        # Input Torque Matrix τ (assumed zero external force)
    
    Tau = np.array([[0],[0],[tau_input] ]) # External force on the cart 

        # Solve for accelerations: M(q) * q_ddot + C(q, q_dot) * q_dot + G(q)=Tau
    q_ddot = np.linalg.solve(M_q, (Tau - np.dot(C_q_qdot, np.array([[omega1], [omega2], [v]])) - G_q))
    #affiche(tau_input)
    return [omega1, q_ddot[0, 0], omega2, q_ddot[1, 0], v, q_ddot[2, 0]]

# Solve the system using Runge-Kutta (RK45)
t_span = (0, T_max)
y0 = [theta1_0_init, omega1_0, theta2_0_init, omega2_0, x_0, v_0]
t_eval = np.arange(0, T_max, dt)

sol = solve_ivp(metronome_matrix_system, t_span, y0, t_eval=t_eval, method="RK45")

# Extract solutions
time = sol.t
theta1 = sol.y[0]
theta2 = sol.y[2]
surface_x = sol.y[4]  # Surface displacement

# Plot results
plt.figure(figsize=(10, 6))

# Metronome Oscillations
plt.subplot(2, 1, 2)
plt.plot(time, theta1, label="Metronome 1", color='g')
plt.plot(time, theta2, label="Metronome 2", color='b')
#plt.xlim(22,30)
#plt.xlim(5,11)
plt.xlim(1,6)
plt.xlabel("Time (s)")
plt.ylabel("Angular Displacement (rad)")
plt.title("Metronome Oscillations")
plt.legend()
plt.grid()

# Surface Movement
#plt.subplot(2, 1, 2)
#plt.xlim(0,15)
#plt.ylim(-1.5,1.5)
#plt.plot(time, surface_x, label="Surface Displacement", color='r')
plt.xlabel("Time (s)")
plt.ylabel("Surface Displacement (m)")
plt.title("Surface Vibration Due to Metronomes")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


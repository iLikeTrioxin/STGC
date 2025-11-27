import casadi as ca
import numpy as np
import do_mpc
import matplotlib.pyplot as plt

# Constants
GAS_CONSTANT = 8.31446261815324   # ideal gas constant J/mol/K

# geometry (from your hardware; change if needed)
TUBE_INNER = 0.026         # m (inner diameter)
TUBE_LENGTH = 0.8             # m total length
TUBE_AREA = np.pi * TUBE_INNER**2
TUBE_VOLUME = TUBE_AREA * TUBE_LENGTH
FILLER_POROSITY = 0.6
GAS_VOLUME = TUBE_VOLUME * FILLER_POROSITY
N_ZONES = 6 # 4x heated zones(100m) + 2x heat exchange zone (200m)
ZONE_VOLUMES = [
    TUBE_AREA * 0.2,
    TUBE_AREA * 0.1,
    TUBE_AREA * 0.1,
    TUBE_AREA * 0.1,
    TUBE_AREA * 0.1,
    TUBE_AREA * 0.2,
]

AIR_DENSITY_AMBIENT = 1.2           # kg/m3 (gas)
AIR_SPECIFIC_HEAT_CAPACITY = 1000.0         # J/(kg K)
dH_SO3 = -197e3     # J/mol per mol SO3 formed (-197 kJ/mol)
A = 1e6            # pre-exponential (units depend on rate law)
E_A = 8e4             # activation energy J/mol
GAS_WALL_HEAT_TRANSFER = 50.0   # W/K (gas-wall coupling per zone)
WALL_HEAT_CAPACITY = 200.0  # J/K (wall thermal capacitance per zone)
AMBIENT_LOSS = 1.0       # W/K (wall loss to ambient)

K_0 = 273.15


model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Inputs
Q = [model.set_variable(var_type='_u', var_name=f'Q{i+1}') for i in range(4)]  # heater powers (W)
FO =  model.set_variable(var_type='_u', var_name= 'FO')  # oxygen volumetric flow m3/s (controlled via restrictor / MAF)
FB =  model.set_variable(var_type='_u', var_name= 'FB')  # burner volumetric flow m3/s (controlled via restrictor / MAF)
Fr = FO + FB

C_SO2_in = model.set_variable('_p', 'C_SO2_in')
C_O2_in  = model.set_variable('_p', 'C_O2_in')
T_in     = model.set_variable('_tvp', 'T_in')
T_amb    = model.set_variable('_tvp', 'T_amb')

# per zone states
Tw    = [model.set_variable('_x', f'Tw_{i}') for i in range(N_ZONES)] # (K) wall temperature
Tg    = [model.set_variable('_x', f'Tg_{i}') for i in range(N_ZONES)] # (K) gas temperature
C_so3 = [model.set_variable('_x', f'C_so3_{i}') for i in range(N_ZONES)] # (mol/m3)
C_so2 = [model.set_variable('_x', f'C_so2_{i}') for i in range(N_ZONES)] # (mol/m3)
C_o2  = [model.set_variable('_x', f'C_o2_{i}') for i in range(N_ZONES)]  # (mol/m3)


ADA = AIR_DENSITY_AMBIENT
ASHC = AIR_SPECIFIC_HEAT_CAPACITY
GWHT = GAS_WALL_HEAT_TRANSFER
WHC = WALL_HEAT_CAPACITY

# r * 0.2m = mol/s
# mol/m3
# k0 * ca.exp(-A_E / (GAS_CONSTANT * temp)) * so2**2 * o2
# first zone definition
rate = A * ca.exp(-E_A / (GAS_CONSTANT * Tg[0])) * C_so2[0]**2 * C_o2[0]
model.set_rhs('C_so3_0', -(Fr/ZONE_VOLUMES[0]) * C_so3[0] + rate)
model.set_rhs('C_so2_0',  (Fr/ZONE_VOLUMES[0]) * (C_SO2_in - C_so2[0]) - rate)
model.set_rhs('C_o2_0',  (Fr/ZONE_VOLUMES[0]) * (C_O2_in - C_o2[0]) - 0.5*rate)

Tg_dot = (Fr / ZONE_VOLUMES[0]) * (T_in - Tg[0]) \
        + (-dH_SO3 / (ADA * ASHC)) * rate \
        + (GWHT / (ADA * ASHC * ZONE_VOLUMES[0])) * (Tw[0] - Tg[0]) \
        - (0.0 / (ADA * ASHC * ZONE_VOLUMES[0])) * (Tg[0] - T_amb)  # keep zero unless you have gas-to-amb loss
model.set_rhs('Tg_0', Tg_dot)

# wall energy balance
# Cw * dT_w/dt = hA*(T_g - T_w) + Q_heater - UA_w*(T_w - T_amb)
Tw_dot = (GWHT * (Tg[0] - Tw[0]) + 0 - AMBIENT_LOSS * (Tw[0] - T_amb)) / WHC
model.set_rhs('Tw_0', Tw_dot)


for i in range(1, N_ZONES):
    rate = A * ca.exp(-E_A / (GAS_CONSTANT * Tg[i])) * C_so2[i]**2 * C_o2[i]
    model.set_rhs(f'C_so3_{i}', -(Fr/ZONE_VOLUMES[i]) * C_so3[i] + rate)
    model.set_rhs(f'C_so2_{i}',  (Fr/ZONE_VOLUMES[i]) * (C_so2[i-1] - C_so2[i]) - rate)
    model.set_rhs(f'C_o2_{i}',  (Fr/ZONE_VOLUMES[i]) * (C_o2[i-1] - C_o2[i]) - 0.5*rate)
    
    Tg_dot = (Fr / ZONE_VOLUMES[i]) * (Tg[i-1] - Tg[i]) \
             + (-dH_SO3 / (ADA * ASHC)) * rate \
             + (GWHT / (ADA * ASHC * ZONE_VOLUMES[i])) * (Tw[i] - Tg[i]) \
             - (0.0 / (ADA * ASHC * ZONE_VOLUMES[i])) * (Tg[i] - T_amb)  # keep zero unless you have gas-to-amb loss
    model.set_rhs(f'Tg_{i}', Tg_dot)

    # wall energy balance
    # Cw * dT_w/dt = hA*(T_g - T_w) + Q_heater - UA_w*(T_w - T_amb)
    Tw_dot = (GWHT * (Tg[i] - Tw[i]) + 2 - AMBIENT_LOSS * (Tw[i] - T_amb)) / WHC
    model.set_rhs(f'Tw_{i}', Tw_dot)

model.setup()

for z in ZONE_VOLUMES:
    print(f'ZONE VOLUME: {z}')

print(model._x)


#estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.005,
}

simulator.set_param(**params_simulator)

p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()

# function for time-varying parameters
def tvp_fun(t_now):
    tvp_num['T_in'] = K_0 + 50
    tvp_num['T_amb'] = K_0 + 25
    return tvp_num

# uncertain parameters
p_num['C_SO2_in'] = 1
p_num['C_O2_in'] = 1
def p_fun(t_now):
    return p_num

simulator.set_tvp_fun(tvp_fun)
simulator.set_p_fun(p_fun)

simulator.setup()

def molarConc(t, p): # K, Pa -> mol/m3
    if 0 in [t, p]: return 0
    moles = 1/((GAS_CONSTANT * t)/p) 
    return moles


u0 = [
    1, 1, 1, 1, # Q's -Heaters
    0.1, 0.1, # FO, FB - airflow
]

x0 = []
x0 += [K_0+200.0 for i in range(N_ZONES)] # (K) wall temperature
x0 += [K_0+200.0 for i in range(N_ZONES)] # (K) gas temperature
x0 += [molarConc(25, 1e8)*0.0 for i in range(N_ZONES)] # (mol/m3)
x0 += [molarConc(25, 1e8)*0.1 for i in range(N_ZONES)] # (mol/m3)
x0 += [molarConc(25, 1e8)*0.1 for i in range(N_ZONES)]  # (mol/m3)

u0 = np.array(u0).reshape(-1,1)
x0 = np.array(x0).reshape(-1,1)

simulator.x0 = x0

for k in range(50):
    y_next = simulator.make_step(u0)

mpc_graphics = do_mpc.graphics.Graphics(simulator.data)

from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax = plt.subplots(N_ZONES+0, sharex=True, figsize=(16,12))

for i in range(N_ZONES):
    mpc_graphics.add_line(var_type='_x', var_name=f'C_so3_{i}', axis=ax[i])
    mpc_graphics.add_line(var_type='_x', var_name=f'C_so2_{i}', axis=ax[i])
    mpc_graphics.add_line(var_type='_x', var_name=f'C_o2_{i}', axis=ax[i])
    
from matplotlib.animation import FuncAnimation, PillowWriter

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = simulator.data['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

anim.save('anim_CSTR.gif', writer=PillowWriter(fps=5))



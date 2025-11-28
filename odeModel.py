import casadi as ca
import numpy as np
import do_mpc
import matplotlib.pyplot as plt
from casadi import SX

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
ZONE_VOLUMES = SX.sym('ZONES', N_ZONES)
ZONE_VOLUMES[0] = TUBE_AREA * 0.2,
ZONE_VOLUMES[1] = TUBE_AREA * 0.1,
ZONE_VOLUMES[2] = TUBE_AREA * 0.1,
ZONE_VOLUMES[3] = TUBE_AREA * 0.1,
ZONE_VOLUMES[4] = TUBE_AREA * 0.1,
ZONE_VOLUMES[5] = TUBE_AREA * 0.2,


AIR_DENSITY_AMBIENT = 1.2           # kg/m3 (gas)
AIR_SPECIFIC_HEAT_CAPACITY = 1000.0         # J/(kg K)
dH_SO3 = -197e3     # J/mol per mol SO3 formed (-197 kJ/mol)
A = 1000            # pre-exponential (units depend on rate law)
E_A = 8e4             # activation energy J/mol
GAS_WALL_HEAT_TRANSFER = 50.0   # W/K (gas-wall coupling per zone)
WALL_HEAT_CAPACITY = 200.0  # J/K (wall thermal capacitance per zone)
AMBIENT_LOSS = 1.0       # W/K (wall loss to ambient)

K_0 = 273.15

def molarConc(t, p): # K, Pa -> mol/m3
    if 0 in [t, p]: return 0
    moles = 1/((GAS_CONSTANT * t)/p) 
    return moles


model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Inputs
Q = model.set_variable(var_type='_u', var_name=f'Q', shape=(4, 1))  # heater powers (W)
FO =  model.set_variable(var_type='_u', var_name= 'FO')  # oxygen volumetric flow m3/s (controlled via restrictor / MAF)
FB =  model.set_variable(var_type='_u', var_name= 'FB')  # burner volumetric flow m3/s (controlled via restrictor / MAF)
Fr = FO + FB

C_SO2_in = model.set_variable('_p', 'C_SO2_in')
C_O2_in  = model.set_variable('_p', 'C_O2_in')
T_in     = model.set_variable('_tvp', 'T_in')
T_amb    = model.set_variable('_tvp', 'T_amb')

# per zone states
Tw    = model.set_variable('_x', 'Tw'   , (N_ZONES, 1)) # (K) wall temperature
Tg    = model.set_variable('_x', 'Tg'   , (N_ZONES, 1))     # (K) gas temperature
C_SO3 = model.set_variable('_x', 'C_SO3', (N_ZONES, 1))  # (mol/m3)
C_SO2 = model.set_variable('_x', 'C_SO2', (N_ZONES, 1))  # (mol/m3)
C_O2  = model.set_variable('_x', 'C_O2' , (N_ZONES, 1))    # (mol/m3)

Twa = SX.sym('Twa', N_ZONES)
Tga = SX.sym('Tga', N_ZONES)
C_SO3a = SX.sym('C_SO3a', N_ZONES)
C_SO2a = SX.sym('C_SO2a', N_ZONES)
C_O2a = SX.sym('C_O2a', N_ZONES)

rate = A * ca.exp(-E_A / (GAS_CONSTANT * Tg)) * C_SO2**2 * C_O2
ADA = AIR_DENSITY_AMBIENT
ASHC = AIR_SPECIFIC_HEAT_CAPACITY
GWHT = GAS_WALL_HEAT_TRANSFER
WHC = WALL_HEAT_CAPACITY
# r * 0.2m = mol/s
# mol/m3
# k0 * ca.exp(-A_E / (GAS_CONSTANT * temp)) * so2**2 * o2
# first zone definition

C_SO3a[1:] =-(Fr/ZONE_VOLUMES[1:]) * C_SO3[1:] + rate[1:] + (Fr/ZONE_VOLUMES[1:])*C_SO3[0:-1]
C_SO2a[1:] = (Fr/ZONE_VOLUMES[1:]) * (C_SO2[0:-1] - C_SO2[1:]) - rate[1:]
C_O2a [1:] = (Fr/ZONE_VOLUMES[1:]) * (C_O2[0:-1] - C_O2[1:]) - 0.5*rate[1:]

Twa[1:] = (GWHT * (Tg[1:] - Tw[1:]) + 0 - AMBIENT_LOSS * (Tw[1:] - T_amb)) / WHC
Tga[1:] = (Fr / ZONE_VOLUMES[1:]) * (Tg[0:-1] - Tg[1:]) \
        + (-dH_SO3 / (ADA * ASHC)) * rate[1:] \
        + (GWHT / (ADA * ASHC * ZONE_VOLUMES[1:])) * (Tw[1:] - Tg[1:]) \
        - (0.0 / (ADA * ASHC * ZONE_VOLUMES[1:])) * (Tg[1:] - T_amb)

C_SO3a[0] = (-(Fr/ZONE_VOLUMES[0]) * C_SO3[0] + rate[0])
C_SO2a[0] =(  (Fr/ZONE_VOLUMES[0]) * (C_SO2_in - C_SO2[0]) - rate[0])
C_O2a[0] = (  (Fr/ZONE_VOLUMES[0]) * (C_O2_in - C_O2[0]) - rate[0])

Twa[0] = (GWHT * (Tg[0] - Tw[0]) + 0 - AMBIENT_LOSS * (Tw[0] - T_amb)) / WHC
Tga[0] = (Fr / ZONE_VOLUMES[0]) * (T_in - Tg[0]) \
        + (-dH_SO3 / (ADA * ASHC)) * rate[0] \
        + (GWHT / (ADA * ASHC * ZONE_VOLUMES[0])) * (Tw[0] - Tg[0]) \
        - (0.0 / (ADA * ASHC * ZONE_VOLUMES[0])) * (Tg[0] - T_amb)  # keep zero unless you have gas-to-amb loss
# wall energy balance
# Cw * dT_w/dt = hA*(T_g - T_w) + Q_heater - UA_w*(T_w - T_amb)


model.set_rhs('Tw', Twa)
model.set_rhs('Tg', Tga)
model.set_rhs('C_SO3', C_SO3a)
model.set_rhs('C_SO2', C_SO2a)
model.set_rhs('C_O2', C_O2a)


# expressions
rmc = molarConc(25, 1e8)
for i in range(N_ZONES):
    model.set_expression(expr_name=f'Cp_so3_{i}', expr=C_SO3[i]/rmc)
    model.set_expression(expr_name=f'Cp_so2_{i}', expr=C_SO2[i]/rmc)
    model.set_expression(expr_name=f'Cp_o2_{i}', expr=C_O2[i]/rmc)
    model.set_expression(expr_name=f'Tgc_{i}', expr=Tg[i]-K_0)
    model.set_expression(expr_name=f'Twc_{i}', expr=Tw[i]-K_0)


model.setup()
print(model._x)
#estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.05,
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
p_num['C_SO2_in'] = molarConc(25, 1e8)*0.2
p_num['C_O2_in'] = molarConc(25, 1e8)*0.1
def p_fun(t_now):
    return p_num

simulator.set_tvp_fun(tvp_fun)
simulator.set_p_fun(p_fun)

simulator.setup()

u0 = [
    1, 1, 1, 1, # Q's -Heaters
    500*1e-6, 500*1e-6, # FO, FB - airflow
]

x0 = []
x0 += [K_0+50.0 for i in range(N_ZONES)] # (K) wall temperature
x0 += [K_0+50.0 for i in range(N_ZONES)] # (K) gas temperature
x0 += [molarConc(25, 1e8)*0.0 for i in range(N_ZONES)] # (mol/m3)
x0 += [molarConc(25, 1e8)*0.0 for i in range(N_ZONES)] # (mol/m3)
x0 += [molarConc(25, 1e8)*0.2 for i in range(N_ZONES)]  # (mol/m3)

u0 = np.array(u0).reshape(-1,1)
x0 = np.array(x0).reshape(-1,1)

simulator.x0 = x0

for k in range(200):
    y_next = simulator.make_step(u0)

mpc_graphics = do_mpc.graphics.Graphics(simulator.data)

from matplotlib import rcParams

rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax = plt.subplots(N_ZONES+1, sharex=True, figsize=(16,12))

for i in range(N_ZONES):
    mpc_graphics.add_line(var_type='_aux', var_name=f'Cp_so3_{i}', axis=ax[i], label='SO3')
    mpc_graphics.add_line(var_type='_aux', var_name=f'Cp_so2_{i}', axis=ax[i], label='SO2')
    mpc_graphics.add_line(var_type='_aux', var_name=f'Cp_o2_{i}', axis=ax[i], label='O2')

    #label_lines  = mpc_graphics.result_lines['_aux', f'Cp_so3']
    #label_lines += mpc_graphics.result_lines['_aux', f'Cp_so2']
    #label_lines += mpc_graphics.result_lines['_aux', f'Cp_o2']
    ax[i].legend()



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



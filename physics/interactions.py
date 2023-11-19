import numpy as np
import numba as nb
  
@nb.jit(nopython=True)
def calc_force_parts(displacement_vector,mass,other_masses,charge,other_charges,separation_vector
                     ,use_gravity_mask
                     ,use_electric_mask
                     ,use_magnetic_mask
                     ,use_spin_mask
                     ,use_gravity=True
                     ,use_electric=True
                     ,G=1,K=1):
    force = np.zeros(3) #3 vector
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i])) #Prevent division by zero
        rhat = displacement_vector[i]/r
        if use_gravity_mask[i]:
            force += rhat*G*mass*other_masses[i]/r**2
        if use_electric_mask[i]:
            force -= rhat*K*charge*other_charges[i]/r**2
    return force #3 vector

@nb.jit(nopython=True)
def calc_electric_parts(displacement_vector,charge,other_charges,separation_vector,k=1):
    force = np.zeros(3)
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i]))
        rhat = displacement_vector[i]/r
        force += rhat*k*charge*other_charges[i]/r**2
    return force

@nb.jit(nopython=True)
def calc_collision_parts(displacement_vector,mass,other_masses,velocity,other_velocites,radius,other_radii):
    #TODO: Implement this
    pass
            
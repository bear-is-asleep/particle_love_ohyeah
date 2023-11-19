import numpy as np
import numba as nb
  
@nb.jit(nopython=True)
def calc_gravity_parts(displacement_vector,mass,other_masses,separation_vector,G=1):
    force = np.zeros(3) #3 vector
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i])) #Prevent division by zero
        rhat = displacement_vector[i]/r
        force += rhat*G*mass*other_masses[i]/r**2
    return force #3 vector

@nb.jit(nopython=True)
def calc_collision_parts(displacement_vector,mass,other_masses,velocity,other_velocites,radius,other_radii):
    #TODO: Implement this
    pass
            
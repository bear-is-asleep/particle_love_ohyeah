import numpy as np
import numba as nb
import torch
from time import time
  
@nb.jit(nopython=True)
def calc_force_parts_cpu(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                     ,G=1,K=1,k=1):
    force = np.zeros(3) #3 vector
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i])) #Prevent division by zero
        rhat = displacement_vector[i]/r
        
        #Forces
        force += rhat*G*mass*other_masses[i]/r**2 #gravity
        force -= rhat*k*charge*other_charges[i]/r**2 #electric
        force += rhat*K*spin*other_spins[i]*r**2 #spring
    return force #3 vector

def calc_force_parts_gpu(displacement_vector, mass, other_masses, charge, other_charges, spin, other_spins, separation_vector, G=1, K=1, k=1):
    # Convert all inputs to PyTorch tensors if they aren't already
    displacement_vector = torch.tensor(displacement_vector, dtype=torch.float32)
    other_masses = torch.tensor(other_masses, dtype=torch.float32)
    other_charges = torch.tensor(other_charges, dtype=torch.float32)
    other_spins = torch.tensor(other_spins, dtype=torch.float32)
    separation_vector = torch.tensor(separation_vector, dtype=torch.float32)

    # Ensure the tensors are on the same device (e.g., CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    displacement_vector = displacement_vector.to(device)
    other_masses = other_masses.to(device)
    other_charges = other_charges.to(device)
    other_spins = other_spins.to(device)
    separation_vector = separation_vector.to(device)

    # Calculate the norms (r) and normalized displacement vectors (rhat)
    norms = torch.norm(displacement_vector, dim=1)
    r = torch.max(separation_vector, norms)
    rhat = displacement_vector / r.unsqueeze(1)

    # Calculate the forces
    force_gravity = G * mass * other_masses / r**2
    force_electric = k * charge * other_charges / r**2
    force_spring = K * spin * other_spins * r**2

    # Combine the forces and multiply by rhat
    force = rhat * (force_gravity.unsqueeze(1) - force_electric.unsqueeze(1) + force_spring.unsqueeze(1))

    # Sum the forces to get the total force
    force_total = torch.sum(force, dim=0)

    return force_total.cpu().numpy() if device == 'cuda' else force_total.numpy() # Return a NumPy array

def compare_force_parts(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1):
    t0 = time()
    cpu = calc_force_parts_cpu(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1)
    t1 = time()
    gpu = calc_force_parts_gpu(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1)
    t2 = time()
    print('CPU:',cpu)
    print('GPU:',gpu)
    print('Difference:',cpu-gpu)
    print('CPU time:',t1-t0)
    print('GPU time:',t2-t1)
    
    print('Mag dif:',np.linalg.norm(cpu-gpu))
    print('norm to cpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu))
    print('norm to gpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(gpu))
    print('norm to gpu+cpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu+gpu))
    print('norm to cpu-gpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu-gpu))

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
            
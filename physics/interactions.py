import numpy as np
import numba as nb
import torch
from time import time
from . import relativity
  
def compute_gpu_forces(edge_index,distances, rel_velocities,separations, positions, velocities, momenta, masses, charges, spins,
                       dts,
                       G=1, K=1, k=1, c=1):
    """
    Compute the forces between particles using GPU acceleration.

    Args:
        edge_index (torch.Tensor): Tensor of shape (2, M) representing the edges between particles.
        distances (torch.Tensor): Tensor of shape (M, 3) representing the distances between particles.
        rel_velocities (torch.Tensor): Tensor of shape (M, 3) representing the relative velocities between particles.
        separations (torch.Tensor): Tensor of shape (M) representing the separations between particles.
        positions (torch.Tensor): Tensor of shape (N, 3) representing the positions of particles.
        velocities (torch.Tensor): Tensor of shape (N, 3) representing the velocities of particles.
        momenta (torch.Tensor): Tensor of shape (N, 3) representing the momenta of particles.
        masses (torch.Tensor): Tensor of shape (N,) representing the masses of particles.
        charges (torch.Tensor): Tensor of shape (N,) representing the charges of particles.
        spins (torch.Tensor): Tensor of shape (N,) representing the spins of particles.
        dt (float): Time step for the simulation.
        G (float, optional): Gravitational constant. Defaults to 1.
        K (float, optional): Coulomb constant. Defaults to 1.
        k (float, optional): Spring constant. Defaults to 1.
        c (float, optional): Speed of light. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the forces acting on particles.
    """
    # Perform Lorentz transformations
    _, distances = relativity.compute_boost(dts, distances, rel_velocities, c=c)
    #print('distances in func:',distances)
    

    # Calculate the norms (r) and normalized displacement vectors (rhat)
    rhats = -distances / torch.norm(distances, dim=1).view(-1,1)
    distance_squares = torch.sum(distances ** 2, dim=1)
    separation_squares = separations ** 2
    rs = torch.max(separation_squares, distance_squares)  # Set r = max(r, |r_i - r_j|) to avoid division by zero

    # Compute electric field
    #torch.cross(positions, velocities)

    # Compute gravitational field contributions and sum them up
    masses = masses[edge_index][0].view(-1,1)
    other_masses = masses[edge_index][1].view(-1,1)
    forces = G * masses*other_masses / rs.view(-1,1)**2 * rhats
    
    #Convert forces back to shape (N,3)
    
    # print('rs: ',rs)
    # print('rhats: ',rhats)
    # print('forces in func:',forces)
    # print('accumulated forces:',accumulate_forces(edge_index,forces,positions.shape[0]))
    forces = accumulate_edges(edge_index,forces,positions.shape[0])
    return forces,distances

def accumulate_edges(edge_index, edge_values, N):
    # Initialize a tensor to store forces for each particle
    node_values = torch.zeros((N, 3), device=edge_values.device, dtype=edge_values.dtype)

    # Get the source and target nodes for each edge
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # For each edge, add the force to both the source and target nodes
    node_values.scatter_add_(0, source_nodes.unsqueeze(1).expand(-1, 3), edge_values)
    #forces.scatter_add_(0, target_nodes.unsqueeze(1).expand(-1, 3), -edge_forces)

    return node_values
    
    
    
    

# def compute_gpu_forces(distances,separations,positions,velocities,momenta,masses,charges,spins,
#                        dt,
#                        G=1,K=1,k=1,c=1):
    
#     print('distances,separations,positions,velocities,momenta,masses,charges,spins')
#     print(distances,separations,positions,velocities,momenta,masses,charges,spins)
#     #Perfom lorentz transformations
#     dts,distances = relativity.compute_boost(dt,distances,velocities,c=1)
#     _,velocities = relativity.compute_boost(dt,velocities,velocities,c=1)
    
#     rhats = distances / torch.norm(distances, dim=1).unsqueeze(1)
#     distance_squares = torch.sum(distances ** 2,dim=1)
#     separation_squares = separations ** 2
#     rs = torch.max(separation_squares, distance_squares)  # Set r = max(r, |r_i - r_j|) to avoid division by zero
#     #rs = torch.where(mask, rs, torch.tensor(float('inf')))
#     # print('rs',rs.shape)
#     # print('masses',masses.shape)
#     # print('distances',distances.shape)
#     # print('mask',mask.shape)
#     # print('distances_squared',distance_squares.shape)
#     # print('separations_squared',separation_squares.shape)
    
#     #Compute electric field
#     torch.cross(positions,velocities)

#     # Compute gravitational field contributions and sum them up
#     f= -G * (masses.unsqueeze(1) / rs.unsqueeze(1)) * rhats
#     force = f.sum(dim=1)
#     return force

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

def calc_gravity_potential(r,mass,G=1):
    return -G*mass/r
def calc_electric_potential(r,charge,k=1):
    return k*charge/r
def calc_spring_potential(r,spin,K=1):
    return K*spin*r

def calc_force_parts_gpu(displacement_vector, mass, other_masses, charge, other_charges, spin, other_spins, separation_vector, G=1, K=1, k=1,device='cpu'):
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
            
import torch
import numpy as np

def calc_gamma(v,c=1):
    """
    Calculate the Lorentz factor gamma for a velocity v.
    """
    gamma = 1/(1-(torch.norm(v,dim=1)/c)**2)**0.5
    return gamma

def compute_boost(x0,xi,v,c=1):
    """
    Compute the boost from a velocity v to a velocity v' in time dt.
    """
    gamma = calc_gamma(v,c=c).view(-1,1)
    beta = v/c
    vnorm = torch.norm(beta,dim=1).unsqueeze(1)
    n = beta/torch.norm(beta,dim=1).unsqueeze(1)
    ndotxi = torch.einsum('ij,ij->i', n,xi).view(-1,1)
    
    # print('gamma in compute_boost',gamma)
    # print('beta in compute_boost',beta)
    # print('n in compute_boost',n)
    # print('ndotxi in compute_boost',ndotxi)
    
    #https://en.wikipedia.org/wiki/Lorentz_transformation
    x0prime = gamma*(x0-vnorm*ndotxi/c**2)
    xiprime = xi+(gamma-1)*ndotxi*n + gamma*vnorm*n
    return x0prime.unsqueeze(1),xiprime

def compute_rel_vel(vi,vj,gammai,c=1):
    """
    Compute the relative velocity of particle j with respect to particle i.
    """
    #https://en.wikipedia.org/wiki/Relative_velocity#Parallel_velocities
    rel_vel = 1/(gammai*(1-np.dot(vi,vj)/c**2))*(vj-vi+vi*(gammai-1)*(np.dot(vi,vj)/np.linalg.norm(vi)**2-1))
    return rel_vel

def compute_rel_vel_tt(edge_index,velocities,c=1):
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    
    vi = velocities[source_nodes]
    vj = velocities[target_nodes]
    gammai = calc_gamma(vi,c=c).view(-1,1)
    vivj = torch.einsum('ij,ij->i', vi,vj).view(-1,1)
    vi2 = torch.norm(vi,dim=1).view(-1,1)**2

    rel_vel = 1/(gammai*(1-vivj/c**2))*(vj-vi+vi*(gammai-1)*(vivj/vi2-1))
    return rel_vel

def compute_distances_tt(edge_index,positions):
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    
    xi = positions[source_nodes]
    xj = positions[target_nodes]
    distances = xj-xi
    return distances
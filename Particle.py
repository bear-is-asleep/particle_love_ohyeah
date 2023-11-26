import numpy as np
import torch
from torch_geometric.data import Data

#My imports
from globals.physics import *
from physics import interactions
from physics import relativity

class Particle(object):
    def __init__(self,id,class_id,position,velocity,mass,charge,spin
            ,c=1
            ,radius=1
            ,color=None
            ,marker='o'
            ,n_trail_points=0):
        #Simulation properties
        self.id = id
        self.class_id = class_id
        self.color = color
        self.marker = marker
        self.n_trail_points = n_trail_points
        self.position = np.array(position)
        if self.n_trail_points > 0:
            self.trail = np.array([self.position]*self.n_trail_points)
    def update_trail(self):
        self.trail[:-1] = self.trail[1:]
        self.trail[-1] = self.position
    def assign_grid_id(self,grid):
        self.grid_id = grid.get_id(self.position)

class RelativisticParticle(Particle):
    def __init__(self,id,class_id,position,velocity,mass,charge,spin
                 ,c=1
                 ,radius=1
                 ,color=None
                 ,marker='o'
                 ,n_trail_points=0):
        super().__init__(id,class_id,position,velocity,mass,charge,spin
                    ,c=c
                    ,radius=radius
                    ,color=color
                    ,marker=marker
                    ,n_trail_points=n_trail_points)
        
        #Particle properties
        self.velocity = np.array(velocity)
        self.gamma = 1/(1-(np.linalg.norm(self.velocity)/c)**2)**0.5
        self.momentum = self.velocity*mass*self.gamma
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.radius = radius
    def __str__(self):
        return f'Relativistic particle {self.id} : x={self.position}, v={self.velocity}, m={self.mass}, q={self.charge}, s={self.spin}, r={self.radius}, trail={self.n_trail_points}'
    def update_state(self,force,velocity,position):
        #self.velocity = interactions.calc_collision_parts(displacement_vector,self.mass,velocity_vector,self.radius)
        self.force = force
        self.velocity = velocity #already computed when updating the force
        self.position = position #already computed
        if self.n_trail_points > 0:
            self.update_trail()
    

class ClassicalParticle(Particle):
    def __init__(self,id,class_id,position,velocity,mass,charge,spin
                 ,c=1
                 ,radius=1
                 ,color=None
                 ,marker='o'
                 ,n_trail_points=0):
        super().__init__(id,class_id,position,velocity,mass,charge,spin,c=c,radius=radius,color=color,marker=marker,n_trail_points=n_trail_points)
        
        #Particle properties
        self.velocity = np.array(velocity,dtype=float)
        self.momentum = self.velocity*mass
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.radius = radius
    def __str__(self):
        return f'Classical Particle {self.id} : x={self.position}, v={self.velocity}, m={self.mass}, q={self.charge}, s={self.spin}, r={self.radius}, trail={self.n_trail_points}'
    def update_force(self,displacement_vector,mass_vector,charge_vector,spin_vector,radius_vector,separation_vector
                     ,G=1,K=1,k=1,use_cpu=True,device='cpu'):
        self.force = np.zeros(3)
        #print(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector)
        if use_cpu:
            self.force += interactions.calc_force_parts_cpu(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector
                                                    ,G=G
                                                    ,K=K
                                                    ,k=k)
        else:
            self.force += interactions.calc_force_parts_gpu(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector
                                                    ,G=G
                                                    ,K=K
                                                    ,k=k
                                                    ,device=device)
    def update_state(self,dt,displacement_vector,velocity_vector,use_cpu=True,device='cpu'):
        #self.velocity = interactions.calc_collision_parts(displacement_vector,self.mass,velocity_vector,self.radius)
        self.velocity += self.force/self.mass*dt
        self.position += self.velocity*dt
        if self.n_trail_points > 0:
            self.update_trail()

class ParticleGraph:
    def __init__(self,particles,c=1):
        self.num_nodes = len(particles)
        self.edge_mask = torch.full((self.num_nodes, self.num_nodes),False, dtype=torch.bool)
        self.set_edges(particles,c=c)
        self.set_nodes(particles)
        self.set_masks()
        self.data = Data(x=self.node_attr, edge_index=self.edge_index, edge_attr=self.edge_attr, mask=self.edge_mask)
    def __str__(self):
        return f'ParticleGraph: {self.data}'
    def set_edges(self,particles,c):
        edge_index = []
        edge_attr = []
        for i, particle_i in enumerate(particles):
            for j, particle_j in enumerate(particles):
                if i != j:# and particle_i.grid_id == particle_j.grid_id:
                    edge_index.append((i, j))
                    r = np.linalg.norm(particle_i.position - particle_j.position)
                    separation = np.array(particle_i.radius + particle_j.radius)
                    if r < separation: #colliding particles, so don't perform field calculations
                        self.edge_mask[i, j] = False
                        self.edge_mask[j, i] = False
                    vj = particle_j.velocity
                    vi = particle_i.velocity
                    gammai = particle_i.gamma
                    #https://en.wikipedia.org/wiki/Relative_velocity#Parallel_velocities
                    rel_vel = 1/(gammai*(1-np.dot(vi,vj)/c**2))*(vj-vi+vi*(gammai-1)*(np.dot(vi,vj)/np.linalg.norm(vi)**2-1))
                    edge_attr.append(list(particle_i.position - particle_j.position) + list(rel_vel) \
                        +[particle_i.radius + particle_j.radius])
                    
        edge_index = list(set(edge_index))
        edge_attr = np.array(edge_attr)
        #Store the edge attributes and indices
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    def set_nodes(self,particles):
        node_attr = []
        for particle in particles:
            node_attr.append([particle.id,particle.grid_id,particle.class_id]+list(particle.position) + list(particle.velocity) + list(particle.momentum) + [particle.mass,particle.charge,particle.spin])
        self.node_attr = np.array(node_attr)
        self.node_attr = torch.tensor(self.node_attr, dtype=torch.float)   
    def set_masks(self):
        for ind,(i, j) in enumerate(self.edge_index.t()):
            self.edge_mask[i, j] = True
            self.edge_mask[j, i] = True
        #self.non_interacting = self.edge_index[:, ~edge_mask[self.edge_index[0], self.edge_index[1]]] #2,M - matched
    def update(self,node_attr,edge_attr,edge_index):
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index
    def show_edges(self,ax,**pltkwargs):
        lines = ()
        for i, j in self.edge_index.t().tolist():
            x = [self.node_attr[i, 0], self.node_attr[j, 0]]
            y = [self.node_attr[i, 1], self.node_attr[j, 1]]
            z = [self.node_attr[i, 2], self.node_attr[j, 2]]
            lines.append((x,y,z))
        ax.plot(lines,**pltkwargs)
    


        
    
    
    
    
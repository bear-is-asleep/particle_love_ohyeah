import numpy as np

#My imports
from globals.physics import *
from physics import interactions

class Particle(object):
    def __init__(self,id,class_id,position,velocity,mass,charge,spin,radius=25
                 ,color=None
                 ,marker='o'
                 ,n_trail_points=0):
        #Simulation properties
        self.id = id
        self.class_id = class_id
        self.color = color
        self.marker = marker
        self.n_trail_points = n_trail_points
        
        #Particle properties
        self.position = np.array(position,dtype=float)
        if self.n_trail_points > 0:
            self.trail = np.array([self.position]*self.n_trail_points)
        self.velocity = np.array(velocity,dtype=float)
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.radius = radius
    def __str__(self):
        return f'Particle {self.id} : x={self.position}, v={self.velocity}, m={self.mass}, q={self.charge}, s={self.spin}, r={self.radius}, trail={self.n_trail_points}'
    def update_force(self,displacement_vector,mass_vector,charge_vector,spin_vector,radius_vector,separation_vector
                     ,G=1,K=1,k=1,use_cpu=True):
        self.force = np.zeros(3)
        if use_cpu:
            self.force += interactions.calc_force_parts_cpu(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector
                                                    ,G=G
                                                    ,K=K
                                                    ,k=k)
        else:
            self.force += interactions.calc_force_parts_gpu(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector
                                                    ,G=G
                                                    ,K=K
                                                    ,k=k)
        # interactions.compare_force_parts(displacement_vector,self.mass,mass_vector,self.charge,charge_vector,self.spin,spin_vector,separation_vector
        #                                             ,G=G
        #                                             ,K=K
        #                                             ,k=k)
    def update_state(self,dt,displacement_vector,velocity_vector,use_cpu=True):
        #self.velocity = interactions.calc_collision_parts(displacement_vector,self.mass,velocity_vector,self.radius)
        self.velocity += self.force/self.mass*dt
        self.position += self.velocity*dt
        if self.n_trail_points > 0:
            self.update_trail()
    def update_trail(self):
        self.trail[:-1] = self.trail[1:]
        self.trail[-1] = self.position
    
        
    
    
    
    
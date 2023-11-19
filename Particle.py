import numpy as np

#My imports
from globals.physics import *
from physics import interactions

class Particle(object):
    def __init__(self,id,position,velocity,mass,charge,spin,radius=25
                 ,use_gravity=False
                 ,use_electric=False
                 ,use_magnetic=False
                 ,use_spin=False
                 ,use_collisions=False):
        #Simulation properties
        self.id = id
        
        #Particle properties
        self.position = np.array(position,dtype=float)
        self.velocity = np.array(velocity,dtype=float)
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.radius = radius
        
        #Which forces to use
        self.use_gravity = use_gravity
        self.use_electric = use_electric
        self.use_magnetic = use_magnetic
        self.use_spin = use_spin
        self.use_collisions = use_collisions
    def __str__(self):
        return f'Particle {self.id} : x={self.position}, v={self.velocity}, m={self.mass}, q={self.charge}, s={self.spin}, r={self.radius}'
    def update_force(self,displacement_vector,mass_vector,charge_vector,spin_vector,radius_vector,separation_vector):
        self.force = np.zeros(3)
        if self.use_gravity:
            self.force += interactions.calc_gravity_parts(displacement_vector,self.mass,mass_vector,separation_vector)
    def update_state(self,timestep,displacement_vector,velocity_vector):
        # if self.use_collisions:
        #     self.velocity = interactions.calc_collision_parts(displacement_vector,self.mass,velocity_vector,self.radius)
        self.velocity += self.force/self.mass*timestep
        self.position += self.velocity*timestep
    
        
    
    
    
    
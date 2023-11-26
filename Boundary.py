import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

class Boundary:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, type='reflective'):
        #set limits
        self.x_min = x_min
        self.x_max = x_max 
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        
        #set boundary type
        self.type = type
        self.check_boundary_type(self.type)
    def __str__(self):
        return f'Boundary: x=[{self.x_min},{self.x_max}], y=[{self.y_min},{self.y_max}], z=[{self.z_min},{self.z_max}] type={self.type}'
    def check_boundary_type(self, type):
        if type not in ['reflective','passive']:
            raise ValueError(f'Boundary type {type} not supported')
    def update_velocities(self, particles):
        if self.type == 'reflective':
            for particle in particles:
                if particle.position[0] - particle.radius < self.x_min or particle.position[0] + particle.radius > self.x_max:
                    particle.velocity[0] *= -1
                    #print('reflecting x: ',particle.position,particle.velocity)
                if particle.position[1] - particle.radius < self.y_min or particle.position[1] + particle.radius > self.y_max:
                    particle.velocity[1] *= -1
                    #print('reflecting y: ',particle.position,particle.velocity)
                if particle.position[2] - particle.radius < self.z_min or particle.position[2] + particle.radius > self.z_max:
                    particle.velocity[2] *= -1
                    #print('reflecting z: ',particle.position,particle.velocity)
    def make_3dpatches(self,**kwargs):
                # Define vertices for each rectangle face of the cuboid
        v0 = [self.x_min, self.y_min, self.z_min]
        v1 = [self.x_min, self.y_max, self.z_min]
        v2 = [self.x_max, self.y_max, self.z_min]
        v3 = [self.x_max, self.y_min, self.z_min]
        v4 = [self.x_min, self.y_min, self.z_max]
        v5 = [self.x_min, self.y_max, self.z_max]
        v6 = [self.x_max, self.y_max, self.z_max]
        v7 = [self.x_max, self.y_min, self.z_max]

        # Create the vertices for the six faces of the cuboid
        faces = [
            [v0, v1, v2, v3],  # Bottom face
            [v4, v5, v6, v7],  # Top face
            [v0, v1, v5, v4],  # Left face
            [v3, v2, v6, v7],  # Right face
            [v1, v2, v6, v5],  # Front face
            [v0, v3, v7, v4],  # Back face
        ]
        return Poly3DCollection(faces,**kwargs)
        
        
            
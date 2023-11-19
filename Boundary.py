import numpy as np

class Boundary:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max 
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def update_velocities(self, particles):
        for particle in particles:
            if particle.position[0] - particle.radius < self.y_min or particle.position[0] + particle.radius > self.y_max:
                particle.velocity[0] *= -1
            if particle.position[1] - particle.radius < self.y_min or particle.position[1] + particle.radius > self.y_max:
                particle.velocity[1] *= -1
            if particle.position[2] - particle.radius < self.z_min or particle.position[2] + particle.radius > self.z_max:
                particle.velocity[2] *= -1
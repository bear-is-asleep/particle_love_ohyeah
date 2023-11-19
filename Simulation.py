from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from time import time

#My imports
from utils import ani_writers
from utils import plotters
from physics import geometry

class Simulation:
    def __init__(self, particles, boundary,timestep,animate_every=1):
        self.particles = particles
        self.boundary = boundary
        self.timestep = timestep
        self.fig, self.ax = plt.subplots()
        # Set style
        self.fig.patch.set_facecolor('black')
        plotters.set_style(self.ax, facecolor='black', axis_off=True)
        self.ani = None
        self.animate_every = animate_every #Only update the animation every n frames

    def init_animation(self):
        # Set plotting parameters
        msize='radius'
        mcolor='charge'
        cmap='viridis'
        # Set up the figure, axis, and plot elements for animation
        particles_data = []
        if cmap is not None:
            if mcolor == 'charge':
                max_mc = max([p.charge for p in self.particles])
                min_mc = min([p.charge for p in self.particles])
            elif mcolor == 'spin':
                max_mc = max([p.spin for p in self.particles])
                min_mc = min([p.spin for p in self.particles])
            elif mcolor == 'mass':
                max_mc = max([p.mass for p in self.particles])
                min_mc = min([p.mass for p in self.particles])
        for i,p in enumerate(self.particles):
            #SET SIZE
            if isinstance(msize, int) or isinstance(msize, float):
                ms = msize
            elif msize == 'mass':
                ms = p.mass
            elif msize == 'charge':
                ms = p.charge
            elif msize == 'radius':
                ms = (p.radius)*2
            else:
                ms = 1
            #Set color
            if mcolor == 'charge':
                mc = p.charge
            elif mcolor == 'spin':
                mc = p.spin
            elif mcolor == 'mass':
                mc = p.mass
            else:
                mc = 'red'
            if cmap is not None:
                mc = plotters.map_value_to_color(mc, min_val=min_mc, max_val=max_mc, cmap=cmap)  
            particles_data.append(self.ax.plot([], [], 'o', ms=ms, color=mc)[0])
        #Initialize the boundary
        self.ax.plot([self.boundary.x_min, self.boundary.x_max], [self.boundary.y_min, self.boundary.y_min], color='white', linestyle='-')
        self.ax.plot([self.boundary.x_min, self.boundary.x_max], [self.boundary.y_max, self.boundary.y_max], color='white', linestyle='-')
        self.ax.plot([self.boundary.x_min, self.boundary.x_min], [self.boundary.y_min, self.boundary.y_max], color='white', linestyle='-')
        self.ax.plot([self.boundary.x_max, self.boundary.x_max], [self.boundary.y_min, self.boundary.y_max], color='white', linestyle='-')
        
        self.particles_data = particles_data
    def update_particles(self):
        t0 = time()
        # Get relation vectors
        displacement_cache = geometry.get_displacement_cache(np.array([p.position for p in self.particles]))
        
        # Animation function called sequentially
        for i,particle in enumerate(self.particles):
            displacement_vector = displacement_cache[i]
            mass_vector = np.array([p.mass for p in self.particles])
            charge_vector = [p.charge for p in self.particles]
            spin_vector = [p.spin for p in self.particles]
            radius_vector = [p.radius for p in self.particles]
            velocity_vector = np.array([p.velocity for p in self.particles])
            separation_vector = np.array([p.radius+particle.radius for p in self.particles])
            particle.update_force(displacement_vector, mass_vector, charge_vector, spin_vector, radius_vector, separation_vector)
        for i,particle in enumerate(self.particles):
            particle.update_state(self.timestep, displacement_vector, velocity_vector)
        self.boundary.update_velocities(self.particles)
        t1 = time()
        print(f'-Physics took {t1-t0:.3f} seconds')

    def animate(self, frame):
        print(f'Frame {frame}')
        
        # Update particles
        for i in range(self.animate_every):
            self.update_particles()
        t1 = time()
        for data, particle in zip(self.particles_data, self.particles):
            data.set_data(particle.position[0], particle.position[1])
        t2 = time()
        print(f'-Animation took {t2-t1:.3f} seconds')
        return self.particles_data

    def run(self, frames=100, interval=50, **kwargs):
        self.init_animation()
        # Run the animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, blit=True)
        plt.show()

    def save(self, filename, fps=30, bitrate=1800, **run_kwargs):
        # Save the animation to an mp4 or gif file
        if self.ani is None:
            self.run(run_kwargs)
        ani_writers.save_animation(self.ani, filename, fps=fps, bitrate=bitrate)
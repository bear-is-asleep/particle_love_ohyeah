from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from time import time
import pandas as pd
import os

#My imports
from utils import ani_writers
from utils import plotters
from physics import geometry

class Simulation:
    def __init__(self, particles, boundary,timestep
                 ,store_values=False
                 ,save_dir='trash'
                 ,animate_every=1):
        self.particles = particles
        self.boundary = boundary
        self.timestep = timestep
        self.fig, self.ax = plt.subplots()
        # Set style
        self.fig.patch.set_facecolor('black')
        plotters.set_style(self.ax, facecolor='black', axis_off=True)
        self.ani = None
        self.animate_every = animate_every #Only update the animation every n frames
        self.store_values = store_values
        self.save_dir = save_dir
        
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
            if p.color is not None:
                mc = p.color
            particles_data.append(self.ax.plot([], [], 'o', ms=ms, color=mc)[0])
        #Initialize the boundary
        if self.boundary is not None :
            if self.boundary.type in ['reflective']:
                color = 'white'
            elif self.boundary.type in ['passive']:
                color = 'black'
            self.ax.plot([self.boundary.x_min, self.boundary.x_max], [self.boundary.y_min, self.boundary.y_min], color=color, linestyle='-')
            self.ax.plot([self.boundary.x_min, self.boundary.x_max], [self.boundary.y_max, self.boundary.y_max], color=color, linestyle='-')
            self.ax.plot([self.boundary.x_min, self.boundary.x_min], [self.boundary.y_min, self.boundary.y_max], color=color, linestyle='-')
            self.ax.plot([self.boundary.x_max, self.boundary.x_max], [self.boundary.y_min, self.boundary.y_max], color=color, linestyle='-')
        
        self.particles_data = particles_data
    def init_particle_history(self,updates):
        os.makedirs(f'{self.save_dir}/particles',exist_ok=True)
        self.particle_history_dfs = [pd.DataFrame(columns=['x','y','z','vx','vy','vz']) for p in self.particles]
        self.particle_save_names = [f'{self.save_dir}/particles/p{p.id}.csv' for p in self.particles]
        for i,df in enumerate(self.particle_history_dfs):
            df['x'] = np.zeros(updates+1)
            df['y'] = np.zeros(updates+1)
            df['z'] = np.zeros(updates+1)
            df['vx'] = np.zeros(updates+1)
            df['vy'] = np.zeros(updates+1)
            df['vz'] = np.zeros(updates+1)
            #First row
            df.iloc[0] = [self.particles[i].position[0],self.particles[i].position[1],self.particles[i].position[2]
                          ,self.particles[i].velocity[0],self.particles[i].velocity[1],self.particles[i].velocity[2]]
            
        
        
            
    def update_particle_history(self):
        for i,p in enumerate(self.particles):
            self.particle_history_dfs[i].iloc[self.updates] = [p.position[0],p.position[1],p.position[2],p.velocity[0],p.velocity[1],p.velocity[2]]
    def save_particle_history(self):
        #Delete rows with all zeros
        self.particle_history_dfs = [df.loc[(df!=0).any(axis=1)] for df in self.particle_history_dfs]
        [df.to_csv(f'{self.save_dir}/particles/p{self.particles[i].id}.csv',index=False) for i,df in enumerate(self.particle_history_dfs)]
    def save_particle_info(self):
        self.particles_info_df = pd.DataFrame(columns=['id','mass','charge','spin','radius'])
        self.particles_info_df['id'] = [p.id for p in self.particles]
        self.particles_info_df['mass'] = [p.mass for p in self.particles]
        self.particles_info_df['charge'] = [p.charge for p in self.particles]
        self.particles_info_df['spin'] = [p.spin for p in self.particles]
        self.particles_info_df['radius'] = [p.radius for p in self.particles]
        self.particles_info_df.to_csv(f'{self.save_dir}/particles_info.csv',index=False)
    def scale_animation(self):
        #Current limits
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        #Get particle positions
        pxmax = max([p.position[0] for p in self.particles])
        pxmin = min([p.position[0] for p in self.particles])
        pymax = max([p.position[1] for p in self.particles])
        pymin = min([p.position[1] for p in self.particles])
        radius = max([p.radius for p in self.particles])
        
        #Scale if needed
        if pxmax > xmax or pxmin < xmin:
            self.ax.set_xlim(pxmin-radius*10,pxmax+radius*10) #times 10 cause i feel like it
        if pymax > ymax or pymin < ymin:
            self.ax.set_ylim(pymin-radius*10,pymax+radius*10) #times 10 cause i feel like it
    def update_particles(self):
        t0 = time()
        # Get relation vectors
        displacement_cache = geometry.get_displacement_cache(np.array([p.position for p in self.particles]))
        mass_vector = np.array([p.mass for p in self.particles])
        charge_vector = np.array([p.charge for p in self.particles])
        spin_vector = np.array([p.spin for p in self.particles])
        radius_vector = np.array([p.radius for p in self.particles])
        velocity_vector = np.array([p.velocity for p in self.particles])
        use_gravity_mask = np.array([p.use_gravity for p in self.particles])
        use_electric_mask = np.array([p.use_electric for p in self.particles])
        use_magnetic_mask = np.array([p.use_magnetic for p in self.particles])
        use_spin_mask = np.array([p.use_spin for p in self.particles])
        
        
        # Animation function called sequentially
        for i,particle in enumerate(self.particles):
            displacement_vector = displacement_cache[i]
            separation_vector = np.array([p.radius+particle.radius for p in self.particles])
            particle.update_force(displacement_vector, mass_vector, charge_vector, spin_vector, radius_vector, separation_vector
                                  , use_gravity_mask, use_electric_mask, use_magnetic_mask, use_spin_mask)
        for i,particle in enumerate(self.particles):
            displacement_vector = displacement_cache[i]
            particle.update_state(self.timestep, displacement_vector, velocity_vector)
        if self.boundary is not None:
            self.boundary.update_velocities(self.particles)
        if self.store_values:
            self.update_particle_history()
            self.updates += 1
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
        self.scale_animation()
        t2 = time()
        print(f'-Animation took {t2-t1:.3f} seconds')
        return self.particles_data

    def run(self, frames=100, interval=50, show_animation=True, **kwargs):
        self.init_animation()
        self.updates=0
        # Run the animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, blit=True)
        if show_animation:
            plt.show()

    def save(self, frames=100,fps=30, bitrate=1800, **run_kwargs):
        os.makedirs(self.save_dir,exist_ok=True)
        if self.store_values:
            self.init_particle_history((1+frames)*(1+self.animate_every))
        # Save the animation to an mp4 or gif file
        if self.ani is None:
            self.run(show_animation=False,frames=frames,**run_kwargs)
        ani_writers.save_animation(self.ani, f'{self.save_dir}/simulation.mp4', fps=fps, bitrate=bitrate)
        if self.store_values:
            self.save_particle_info()
            self.save_particle_history()
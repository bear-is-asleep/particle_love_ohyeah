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
                 ,animate_every=1
                 ,show_trails=False
                 ,G=1
                 ,k=1
                 ,K=1
                 ,use_cpu=True
                 ,compare=False):
        #Set physics constants
        self.G=G
        self.k=k
        self.K=K
        
        #Set simulation properties        
        self.particles = particles
        self.boundary = boundary
        self.timestep = timestep
        
        #Set computational properties
        self.use_cpu = use_cpu
        self.compare = compare
        
        # Set style
        self.fig, self.ax = plt.subplots(figsize=(15,9))
        self.fig.patch.set_facecolor('black')
        self.fig.subplots_adjust(right=0.7) #make room for legend
        plotters.set_style(self.ax, facecolor='black', axis_off=True)
        
        #Set animation properties
        self.ani = None
        self.animate_every = animate_every #Only update the animation every n frames
        self.store_values = store_values
        self.save_dir = save_dir
        self.show_trails = show_trails
        
    def __str__(self):
        return f'Simulation: dt={self.timestep}, show_trails={self.show_trails}, animate_every={self.animate_every}, store_values={self.store_values}'
    def init_animation(self):
        # Set plotting parameters
        msize='radius'
        mcolor='charge'
        cmap='bwr'
        # Set up the figure, axis, and plot elements for animation
        particles_data = []
        if self.show_trails:
            trail_data = []
        class_handles = [] #for legend
        class_ids = [] #get unique class ids
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
                ms = (p.radius)*2
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
            #Set marker
            if p.marker is not None:
                marker = p.marker
            else:
                marker = 'o'
            label = f'Class {p.class_id}: m={p.mass}, q={p.charge}, s={p.spin}, r={p.radius}'
            particles_data.append(self.ax.plot([], [], marker, ms=ms, color=mc,label=label)[0]) #points
            if self.show_trails:
                trail_data.append(self.ax.plot([], [], ms=ms, color=mc, alpha=0.3)[0]) #trails
            if p.class_id not in class_ids:
                class_handles.append(particles_data[i])
                class_ids.append(p.class_id)
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
        #Set up legend
        self.ax.legend(handles=class_handles,bbox_to_anchor=(1, 1))
        self.particles_data = particles_data
        if self.show_trails:
            self.trail_data = trail_data
        self.class_handles = class_handles

    def update_trails(self):
        for i,p in enumerate(self.particles):
            x = p.trail[:,0]
            y = p.trail[:,1]
            self.trail_data[i].set_data(x,y)
    
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
            xbuffer = (pxmax-pxmin)*0.3
            self.ax.set_xlim(pxmin-xbuffer,pxmax+xbuffer) #times 10 cause i feel like it
        if pymax > ymax or pymin < ymin:
            ybuffer = (pymax-pymin)*0.3
            self.ax.set_ylim(pymin-ybuffer,pymax+ybuffer) #times 10 cause i feel like it
    def update_particles(self):
        t0 = time()
        # Get relation vectors
        displacement_cache = geometry.get_displacement_cache(np.array([p.position for p in self.particles]))
        mass_vector = np.array([p.mass for p in self.particles])
        charge_vector = np.array([p.charge for p in self.particles])
        spin_vector = np.array([p.spin for p in self.particles])
        radius_vector = np.array([p.radius for p in self.particles])
        velocity_vector = np.array([p.velocity for p in self.particles])
        
        # update particles
        ta = time()
        for i,particle in enumerate(self.particles):
            displacement_vector = displacement_cache[i]
            separation_vector = np.array([p.radius+particle.radius for p in self.particles])
            particle.update_force(displacement_vector, mass_vector, charge_vector, spin_vector, radius_vector, separation_vector,
                                  G=self.G,K=self.K,k=self.k,use_cpu=self.use_cpu)
            particle.update_state(self.timestep, displacement_vector, velocity_vector, use_cpu=self.use_cpu)
        if self.compare:
            tb = time()
            particles_gpu = self.particles.copy()
            for i,particle in enumerate(particles_gpu):
                displacement_vector = displacement_cache[i]
                separation_vector = np.array([p.radius+particle.radius for p in self.particles])
                particle.update_force(displacement_vector, mass_vector, charge_vector, spin_vector, radius_vector, separation_vector,
                                    G=self.G,K=self.K,k=self.k,use_cpu=not self.use_cpu)
                particle.update_state(self.timestep, displacement_vector, velocity_vector, use_cpu=not self.use_cpu)
        tc = time()
        if self.compare:
            print(f'--Physics update took {tb-ta:.3f} seconds (CPU) and {tc-tb:.3f} seconds (GPU)')
        if self.boundary is not None:
            self.boundary.update_velocities(self.particles)
        if self.store_values:
            self.update_particle_history()
        self.updates += 1
        t1 = time()
        if not self.compare:
            print(f'--Physics update {self.updates} took {t1-t0:.3f} seconds')

    def animate(self, frame):
        print(f'Frame {frame}')
        
        # Update particles
        for _ in range(self.animate_every):
            self.update_particles()
        t1 = time()
        for data, particle in zip(self.particles_data, self.particles):
            data.set_data(particle.position[0], particle.position[1])
        if self.show_trails:
            self.update_trails()
        self.scale_animation()
        t2 = time()
        print(f'-Animation took {t2-t1:.3f} seconds')
        if self.show_trails:
            return self.trail_data + self.particles_data
        return self.particles_data

    def run(self, frames=100, interval=50, show_animation=True, **kwargs):
        self.init_animation()
        self.updates=0
        # Run the animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, blit=True, **kwargs)
        if show_animation:
            plt.show()

    def simulate(self, updates=100, compare=False, **kwargs):
        os.makedirs(self.save_dir,exist_ok=True)
        self.updates=0
        self.init_particle_history(updates)
        for i in range(updates):
            self.update_particles()
        self.save_particle_info()
        self.save_particle_history()

    def save(self, frames=100,fps=30, bitrate=1800, **run_kwargs):
        os.makedirs(self.save_dir,exist_ok=True)
        # Save the animation to an mp4 or gif file
        if self.ani is None:
            self.run(show_animation=False,frames=frames,**run_kwargs)
        ani_writers.save_animation(self.ani, f'{self.save_dir}/simulation.mp4', fps=fps, bitrate=bitrate)
    
    def simulate_and_save(self, frames=100, fps=30, bitrate=1800, **run_kwargs):
        os.makedirs(self.save_dir,exist_ok=True)
        self.updates=0
        self.init_particle_history((1+frames)*(1+self.animate_every))
        # Save the animation to an mp4 or gif file
        if self.ani is None:
            self.run(show_animation=False,frames=frames,**run_kwargs)
        ani_writers.save_animation(self.ani, f'{self.save_dir}/simulation.mp4', fps=fps, bitrate=bitrate)
        self.save_particle_info()
        self.save_particle_history()
        
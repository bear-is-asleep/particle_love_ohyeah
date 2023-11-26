from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from time import time
import pandas as pd
import os
import torch

#My imports
from utils import ani_writers
from utils import plotters
from physics import geometry,fields,interactions,relativity
from globals.maps import *
from Particle import ParticleGraph

class Simulation:
    def __init__(self, particles, boundary,fields,dt
                 ,store_values=False
                 ,save_dir='trash'
                 ,animate_every=1
                 ,show_trails=False
                 ,show_field_ind=None #index of field to show
                 ,compare=False
                 ,use_fields=False
                 ,G=1
                 ,k=1
                 ,K=1
                 ,c=1):
        #Set physics constants
        self.G=G
        self.k=k
        self.K=K
        self.c=c
        
        #Set simulation properties        
        self.particles = particles
        self.boundary = boundary
        self.dt = dt
        self.fields=fields
        self.assign_grid_ids() #assign grid ids to particles
        self.init_particle_graphs() #initialize particle graph
        
        #Set computational properties
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compare = compare
        self.use_fields = use_fields
        
        # Set style
        self.fig = plt.figure(figsize=(15,9))
        self.ax = self.fig.add_subplot(projection='3d')
        self.fig.patch.set_facecolor('black')
        self.fig.subplots_adjust(right=0.7) #make room for legend
        plotters.set_style(self.ax, facecolor='black', axis_off=True)
        
        #Set animation properties
        self.ani = None
        self.animate_every = animate_every #Only update the animation every n frames
        self.store_values = store_values
        self.save_dir = save_dir
        self.show_trails = show_trails
        self.show_field_ind = show_field_ind
        
    def __str__(self):
        return f'Simulation: dt={self.dt}, show_trails={self.show_trails}, animate_every={self.animate_every}, store_values={self.store_values}'
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
        fields_data = [] #for fields
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
            particles_data.append(self.ax.scatter(p.position[0], p.position[1], p.position[2], marker=marker, s=ms, color=mc,label=label)) #points
            if self.show_trails:
                trail_data.append(self.ax.plot([], [], [], ms=ms, color=mc, alpha=0.2)[0]) #trails
            if p.class_id not in class_ids:
                class_handles.append(particles_data[i])
                class_ids.append(p.class_id)
        #Initialize the boundary
        if self.boundary is not None :
            if self.boundary.type in ['reflective']:
                color = 'white'
            elif self.boundary.type in ['passive']:
                color = 'black'
            pc = self.boundary.make_3dpatches(edgecolor=color,alpha=0.1,linewidths=2)
            self.ax.add_collection3d(pc)
            self.ax.set_xlim(self.boundary.x_min,self.boundary.x_max)
            self.ax.set_ylim(self.boundary.y_min,self.boundary.y_max)
            self.ax.set_zlim(self.boundary.z_min,self.boundary.z_max)
        if self.fields is not None and self.show_field_ind is not None:
            fields_data.append(self.fields[self.show_field_ind].show_field(self.ax,cmap='bwr',alpha=0.5,marker='s'))
            if self.fields[self.show_field_ind].divisions <= 25:
                self.fields[self.show_field_ind].show_lines(self.ax,color='white')
        #Set up legend
        self.ax.legend(handles=class_handles,bbox_to_anchor=(1, 1))
        self.particles_data = particles_data
        if self.show_trails:
            self.trail_data = trail_data
        self.class_handles = class_handles
        self.fields_data = fields_data

    def update_trails(self):
        for i,p in enumerate(self.particles):
            p.update_trail()
            x = p.trail[:,0]
            y = p.trail[:,1]
            z = p.trail[:,2]
            self.trail_data[i].set_xdata(x)
            self.trail_data[i].set_ydata(y)
            self.trail_data[i].set_3d_properties(z)
    def update_field_plot(self):
        field = self.fields[self.show_field_ind]
        field_data = self.fields_data[self.show_field_ind]
        if field.dynamic:
            grid = field.grid
            x = grid[:,:,:,1].flatten()
            y = grid[:,:,:,2].flatten()
            z = grid[:,:,:,3].flatten()
            field = grid[:,:,:,4].flatten()
            field_data._offsets3d = (x,y,z)
            field_data.set_array(field)
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
            #self.ax.set_xlim(pxmin-xbuffer,pxmax+xbuffer) #times 10 cause i feel like it
        if pymax > ymax or pymin < ymin:
            ybuffer = (pymax-pymin)*0.3
            #self.ax.set_ylim(pymin-ybuffer,pymax+ybuffer) #times 10 cause i feel like it
            
    def assign_grid_ids(self):
        for i,p in enumerate(self.particles):
            p.assign_grid_id(self.fields[0])
        
    def mask_self_particles(self,particle,to_torch=True):
        if to_torch:
            return torch.tensor([p.id != particle.id for p in self.particles], dtype=torch.bool, device=self.device)
        return np.array([p.id != particle.id for p in self.particles])
    
    def init_particle_graphs(self):
        self.particle_graphs = [None]*len(self.fields)
        self.particle_graphs[FIELD_MAP['gravity']] = ParticleGraph(self.particles,self.c)   
    
    def update_physics(self):
        t0 = time()
        t1 = time()
        print(f'---Getting relation vectors took {t1-t0:.3f} seconds')
        # Update particle positions
        def update_particle_forces(particles):
            t0 = time()
            
            #Unpack particle graph
            edge_index = self.particle_graphs[FIELD_MAP['gravity']].edge_index #2,M - matched
            edge_mask = self.particle_graphs[FIELD_MAP['gravity']].edge_mask #N,N 
            
            #Edge attributes
            edge_attr = self.particle_graphs[FIELD_MAP['gravity']].edge_attr #M,4
            edge_attr = edge_attr[edge_mask[edge_index[0], edge_index[1]]]
            distances = edge_attr[:,DX_IND:DZ_IND+1] #M,3
            rel_velocities = edge_attr[:,RVX_IND:RVZ_IND+1] #M,3
            separations = edge_attr[:,SEP_IND] #M
            dts = relativity.calc_gamma(rel_velocities,c=self.c)*self.dt #N
            
            # print('edge_index: ',edge_index)
            # print('edge_attr: ',edge_attr)
            # print('distances: ',distances)
            # print('rel_velocities: ',rel_velocities)
            # print('separations: ',separations)
            
            #Unpack node attributes
            node_attr = self.particle_graphs[FIELD_MAP['gravity']].node_attr #N,15
            part_ids = node_attr[:,ID_IND] #N
            grid_ids = node_attr[:,GRID_IND] #N
            class_ids = node_attr[:,CLASS_IND] #N 
            positions = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,X_IND:Z_IND+1] #N,3 - total
            velocities = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,VX_IND:VZ_IND+1]  #N,3
            momenta = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,PX_IND:PZ_IND+1] #N,3
            masses = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,M_IND] #N
            charges = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,Q_IND] #N
            spins = self.particle_graphs[FIELD_MAP['gravity']].node_attr[:,S_IND] #N
            gammas = relativity.calc_gamma(velocities,c=self.c) #N
            
            # print('node_attr: ',node_attr)
            # print('part_ids: ',part_ids)
            # print('grid_ids: ',grid_ids)
            # print('class_ids: ',class_ids)
            # print('positions: ',positions)
            # print('velocities: ',velocities)
            # print('momenta: ',momenta)
            # print('masses: ',masses)
            # print('charges: ',charges)
            # print('spins: ',spins)
            # print('gammas: ',gammas)
            # print('dts: ',dts)
            
            t1 = time()
            print(f'----Unpacking particle graph took {t1-t0:.3f} seconds')
            #Compute forces and length contracted distances
            forces,distances = interactions.compute_gpu_forces(edge_index,distances, rel_velocities, separations, positions, velocities, momenta, masses, charges, spins
                       ,dts
                       ,G=self.G, K=self.K, k=self.k, c=self.c)
            momenta += forces*self.dt*gammas.view(-1,1) #p = F*dt*gamma
            velocities += momenta/(masses.view(-1,1)*gammas.view(-1,1)) #v = p/m*gamma
            positions += gammas.view(-1,1)*velocities*self.dt #dx = v*dt*gamma
            rel_velocities = relativity.compute_rel_vel_tt(edge_index,velocities,c=self.c)
            print('forces: ',forces)
            print('momenta: ',momenta)
            print('velocities: ',velocities)
            print('positions: ',positions)
            print('rel_velocities: ',rel_velocities)
            
            t2 = time()
            print(f'----Computing forces took {t2-t1:.3f} seconds')
            
            #Update particle graph
            node_attr[:,X_IND:Z_IND+1] = positions
            node_attr[:,VX_IND:VZ_IND+1] = velocities
            node_attr[:,PX_IND:PZ_IND+1] = momenta
            node_attr[:,GRID_IND] = grid_ids
            
            edge_attr[:,DX_IND:DZ_IND+1] = distances
            edge_attr[:,RVX_IND:RVZ_IND+1] = rel_velocities
            self.particle_graphs[FIELD_MAP['gravity']].node_attr = node_attr
            self.particle_graphs[FIELD_MAP['gravity']].edge_attr = edge_attr
            t3 = time()
            print(f'----Updating particle graph took {t3-t2:.3f} seconds')
            
            #Update particles
            forces = forces.cpu().numpy()
            velocities = velocities.cpu().numpy()
            positions = positions.cpu().numpy()
            part_ids = np.array(part_ids.cpu().numpy(),dtype=int)
            for i in part_ids: #match particle ids to graph ids
                particles[i].update_state(forces[i],velocities[i],positions[i])
                particles[i].assign_grid_id(self.fields[FIELD_MAP['gravity']])
                grid_ids[i] = node_attr[i,GRID_IND] #update grid ids
            #self.assign_grid_ids()
            t4 = time()
            print(f'----Updating particles took {t4-t3:.3f} seconds')
            #TODO: update particle graph's grid ids
            #self.particle_graphs[FIELD_MAP['gravity']].edge_index = edge_index
            
            return particles
            
        def update_particle_fields(particles,fields):   
            t0 = time() 
            for i,field in enumerate(self.fields):
                print(self.particle_graphs[FIELD_MAP['gravity']])
                print(i,field)
                if field.dynamic:
                    field.update(self.particle_graphs[i])
            t1 = time()
            print(f'---Updating fields took {t1-t0:.3f} seconds')
        if not self.use_fields:
            update_particle_forces(self.particles)
        if self.boundary is not None:
            t2 = time()
            self.boundary.update_velocities(self.particles)
            t3 = time()
            print(f'---Reflecting off boundary took {t3-t2:.3f} seconds')
        if self.store_values:
            self.update_particle_history()
        t4 = time()
        t5 = time()
        print(f'---Making particle graphs took {t4-t3:.3f} seconds')
        print(f'---Assigning grid ids took {t5-t4:.3f} seconds')
        self.updates += 1

    def animate(self, frame):
        print(f'Frame {frame}')
        
        # Update particles
        for _ in range(self.animate_every):
            self.update_physics()
        t1 = time()
        for data, particle in zip(self.particles_data, self.particles):
           data._offsets3d = ([particle.position[0]],[particle.position[1]],[particle.position[2]])
        if self.show_field_ind is not None:
           self.update_field_plot()
            
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

    def simulate(self, updates=100, **kwargs):
        os.makedirs(self.save_dir,exist_ok=True)
        self.updates=0
        self.init_particle_history(updates)
        for i in range(updates):
            self.update_physics()
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
        
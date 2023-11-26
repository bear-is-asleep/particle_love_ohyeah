import yaml
import numpy as np
import os


from Particle import RelativisticParticle as Particle
from Boundary import Boundary
from Simulation import Simulation
from Field import Gravity

from globals.maps import FIELD_MAP

#Set seed for reproducibility
np.random.seed(420)

#Set YAML file paths
kingdom_fname = 'config/kingdom.yml'
simulation_fname = 'config/simulation.yml'

def copy_yamls(save_path):
		os.system(f'cp {kingdom_fname} {save_path}')
		os.system(f'cp {simulation_fname} {save_path}')

kingdom_yaml = kingdom_fname
simulation_yaml = simulation_fname

# Read YAML configurations
with open(kingdom_yaml, 'r') as file:
    kingdom_config = yaml.safe_load(file)
with open(simulation_yaml, 'r') as file:
    simulation_config = yaml.safe_load(file)

#Extract simulation properties
sim_config = simulation_config['simulation']
dt = sim_config['dt']
animate_every = sim_config['animate_every']
frames = sim_config['frames']
interval = sim_config['interval']
fps = sim_config['fps']
bitrate = sim_config['bitrate']
store_values = sim_config['store_values']
n_trail_points = sim_config['n_trail_points']
sim_mode = sim_config['mode']
use_fields = sim_config['use_fields']
compare = sim_config['compare']
show_trails = True if n_trail_points > 0 else False

#Extract physics properties
physics_config = simulation_config['physics']
G = np.double(physics_config['G'])
K = np.double(physics_config['K'])
c = np.double(physics_config['c'])
e_0 = np.double(physics_config['e_0'])
mu_0 = np.double(physics_config['mu_0'])
hbar = np.double(physics_config['hbar'])
k = 1/(4*np.pi*e_0)


#Extract boundary properties
boundary = simulation_config['boundary']
box_size = boundary['box_size']
boundary_type = boundary['type']
boundary = Boundary(x_min=-box_size, x_max=box_size, y_min=-box_size, y_max=box_size, z_min=-box_size, z_max=box_size, type=boundary_type)

#Extract field properties
fields = simulation_config['fields']
divisions = fields['divisions']
show_field = fields['show_field']
#Gravity
gravity = fields['gravity']
gravity = Gravity(xmin=-box_size, xmax=box_size, ymin=-box_size, ymax=box_size, zmin=-box_size, zmax=box_size, divisions=divisions, attributes=['id', 'x', 'y', 'z', 'g'], dynamic=gravity['dynamic'])

fields = [gravity]

# Initialize particles
particle_keys = [key for key in kingdom_config.keys() if key[:8] == 'particle']
num_particles = [kingdom_config[key]['n'] for key in particle_keys]
running_particle_count = np.zeros(len(num_particles))
particles = [None]*sum(num_particles)

#Create particles
for i in range(sum(num_particles)):
	for j,n in enumerate(running_particle_count):
		n_limit = num_particles[j]
		if n < n_limit:
			running_particle_count[j] += 1
			particle = kingdom_config[particle_keys[j]]
			mass = particle['mass']
			charge = particle['charge']
			spin = particle['spin']
			radius = particle['radius']
			color = particle['color']
			marker = particle['marker']
			vmax = particle['vmax']
			break
	velocity = np.random.uniform(-1,1,3)
	velocity = velocity/np.linalg.norm(velocity)*vmax #scale to vmax
	position = np.random.uniform(-box_size,box_size,3)
 
	if i == 0: 
		velocity = np.array([0,vmax,0],dtype=float)
		position = np.array([0,500,0],dtype=float)
	elif i == 1: 
		velocity = np.array([0,-vmax,0],dtype=float)
		position = np.array([0,-500,0],dtype=float)
	particles[i] = Particle(id=i
              ,class_id=j
							,position=position
							,velocity=velocity#np.random.uniform(-vmax/2,vmax/2,3) #stay below c
							,mass=mass
							,charge=charge
							,spin=spin
       				,c=c #speed of light
							,radius=radius
							,color=color
							,marker=marker
             	,n_trail_points=n_trail_points)
if sim_mode != 'run':
	for i in range(int(1e6)):
		save_path = f'simulations/{sim_config["name"]}{i}'
		if not os.path.exists(save_path):
			break
else:
	save_path = 'trash'
if show_field is None:
  show_field_ind = None
else:
  show_field_ind = FIELD_MAP[show_field]
#Put the particles in the simulation
sim = Simulation(particles, boundary, fields
				 ,dt=dt
				 ,store_values=store_values
				 ,animate_every=animate_every
				 ,save_dir=save_path
     		 ,show_trails=show_trails
         ,show_field_ind=show_field_ind
         ,compare=compare
         ,use_fields=use_fields
         ,G=G
         ,K=K
         ,k=k
         ,c=c)

if len(particles) < 10:
	[print(particle) for particle in particles]
print(boundary)
print(sim)

if sim_mode == 'run':
  sim.run(frames=frames, interval=interval)
elif sim_mode == 'simulate':
	sim.store_values = True #always store values when simulating
	sim.simulate(updates=frames)
  #Copy yamls to save directory
	copy_yamls(save_path)


elif sim_mode == 'save':
	sim.save(fps=fps,bitrate=bitrate,frames=frames,interval=interval)
	copy_yamls(save_path)
	print(f'Saved to {save_path}')
elif sim_mode == ['simulate','save']:
	sim.store_values = True #always store values when simulating
	sim.simulate_and_save(frames=frames,fps=fps,bitrate=bitrate,interval=interval)
  #Copy yamls to save directory
	copy_yamls(save_path)

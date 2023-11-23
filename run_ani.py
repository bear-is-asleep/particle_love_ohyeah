import yaml
import numpy as np
import os


from Particle import Particle
from Boundary import Boundary
from Simulation import Simulation

#Set seed for reproducibility
#np.random.seed(420)

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
save_path = f'simulations/{sim_config["name"]}'
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

print(type(G),type(K),type(c),type(e_0),type(mu_0),type(hbar),type(k))


#Extract boundary properties
boundary = simulation_config['boundary']
box_size = boundary['box_size']
boundary_type = boundary['type']
boundary = Boundary(x_min=-box_size, x_max=box_size, y_min=-box_size, y_max=box_size, z_min=-box_size, z_max=box_size, type=boundary_type)

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
   
	particles[i] = Particle(id=i
              ,class_id=j
							,position=np.random.uniform(-box_size,box_size,3)
							,velocity=np.random.uniform(-vmax,vmax,3)
							,mass=mass
							,charge=charge
							,spin=spin
							,radius=radius
							,color=color
							,marker=marker
             	,n_trail_points=n_trail_points)

#Put the particles in the simulation
sim = Simulation(particles, boundary
				 ,timestep=dt
				 ,store_values=store_values
				 ,animate_every=animate_every
				 ,save_dir=save_path
     		 ,show_trails=show_trails
         ,G=G
         ,K=K
         ,k=k
         ,use_cpu=True)


[print(particle) for particle in particles]
print(boundary)
print(sim)

if sim_mode == 'simulate':
	sim.store_values = True #always store values when simulating
	sim.simulate(updates=frames)
  #Copy yamls to save directory
	copy_yamls(save_path)
elif sim_mode == 'run':
  sim.run(frames=frames, interval=interval)
	
elif sim_mode == 'save':
	sim.save(fps=fps,bitrate=bitrate,frames=frames,interval=interval)
elif sim_mode == ['simulate','save']:
	sim.store_values = True #always store values when simulating
	sim.simulate_and_save(frames=frames,fps=fps,bitrate=bitrate,interval=interval)
  #Copy yamls to save directory
	copy_yamls(save_path)

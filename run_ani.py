import yaml
import numpy as np

from Particle import Particle
from Boundary import Boundary
from Simulation import Simulation

kingdom_yaml = 'config/kingdom1.yaml'
simulation_yaml = 'config/simulation.yaml'

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
show_trails = sim_config['show_trails']
sim_mode = sim_config['mode']

particle_keys = [key for key in kingdom_config.keys() if key[:8] == 'particle']
boundary = kingdom_config['boundary']

#Extract boundary properties
box_size = boundary['box_size']
boundary_type = boundary['type']
boundary = Boundary(x_min=-box_size, x_max=box_size, y_min=-box_size, y_max=box_size, z_min=-box_size, z_max=box_size, type=boundary_type)

# Create particles
num_particles = [kingdom_config[key]['n'] for key in particle_keys]
running_particle_count = np.zeros(len(num_particles))
particles = [None]*sum(num_particles)

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
			use_gravity = particle['use_gravity']
			use_electric = particle['use_electric']
			use_magnetic = particle['use_magnetic']
			use_spin = particle['use_spin']
			use_collisions = particle['use_collisions']
			n_trail_points = particle['n_trail_points']
			break
   
	particles[i] = Particle(id=i
              ,class_id=j
							,position=np.random.uniform(-box_size,box_size,3)
							,velocity=np.random.uniform(-1,1,3)
							,mass=mass
							,charge=charge
							,spin=spin
							,radius=radius
							,color=color
							,use_gravity=use_gravity
							,use_electric=use_electric
							,use_magnetic=use_magnetic
							,use_spin=use_spin
       						,use_collisions=use_collisions
             				,n_trail_points=n_trail_points)

sim = Simulation(particles, boundary
				 ,timestep=dt
				 ,store_values=store_values
				 ,animate_every=animate_every
     			,show_trails=show_trails)

[print(particle) for particle in particles]
print(boundary)
print(sim)

if sim_mode == 'simulate':
	sim.store_values = True #always store values when simulating
	sim.simulate(updates=frames)
elif sim_mode == 'run':
	sim.run(frames=frames, interval=interval)
elif sim_mode == 'save':
	sim.save(fps=fps,bitrate=bitrate,frames=frames,interval=interval)

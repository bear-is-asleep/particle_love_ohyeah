from sympy.physics import units
import numpy as np

#My imports
from Particle import Particle
from Simulation import Simulation
from Boundary import Boundary

#Constants


#Variables

#Particle properties
mass = 100
charge = 1
spin = 1
radius = 3
use_gravity = True

#Boundary properties
box_size = 1000

#Simulation properties
dt = .1
animate_every = 10
frames = 1
interval = 20
fps = 30
bitrate = 1800


# Setup particles
particles = [ Particle(id=i
                       ,position=np.random.uniform(-box_size,box_size,3)
                       ,velocity=np.random.uniform(-1,1,3)
                       ,mass=mass
                       ,charge=charge
                       ,spin=spin
                       ,radius=radius
                       ,use_gravity=use_gravity) for i in range(1000)
]

# SEtup boundary
boundary = Boundary(x_min=-box_size, x_max=box_size, y_min=-box_size, y_max=box_size, z_min=-box_size, z_max=box_size)

sim = Simulation(particles, boundary, timestep=dt,animate_every=10)
sim.run(frames=frames, interval=interval)
#sim.save('trash/simulation.mp4',fps=fps,bitrate=bitrate,frames=frames,interval=interval)
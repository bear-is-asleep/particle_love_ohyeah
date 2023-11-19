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
charge = 500
spin = 1
radius = 3
use_gravity = False
use_electric = True

#Boundary properties
box_size = 1000
boundary_type = 'reflective'

#Simulation properties
dt = 1
animate_every = 10
frames = 1000
interval = 20
fps = 30
bitrate = 1800
store_values = True


# Setup particles
particles = [ Particle(id=i
                       ,position=np.random.uniform(-box_size,box_size,3)
                       ,velocity=np.random.uniform(-1,1,3)
                       ,mass=mass
                       ,charge=charge if i%2==0 else -charge
                       ,spin=spin
                       ,radius=radius
                       ,color='blue' if i%2==0 else 'red'
                       ,use_gravity=use_gravity
                       ,use_electric=use_electric) for i in range(200)
]

# SEtup boundary
boundary = Boundary(x_min=-box_size, x_max=box_size, y_min=-box_size, y_max=box_size, z_min=-box_size, z_max=box_size
                    ,type=boundary_type)

sim = Simulation(particles, boundary
                 ,timestep=dt
                 ,store_values=store_values
                 ,animate_every=animate_every)
#sim.run(frames=frames, interval=interval)
#sim.save(fps=fps,bitrate=bitrate,frames=frames,interval=interval)
sim.simulate(updates=frames)
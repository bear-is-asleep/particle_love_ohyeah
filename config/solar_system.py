import numpy as np

from Particle import Particle


sun = Particle(id=0
                ,position=np.array([0,0,0])
                ,velocity=np.array([0,0,0])
                ,mass=1.989e30 #kg
                ,charge=0
                ,spin=0
                ,radius=6.957e8 #m
                ,color='yellow')


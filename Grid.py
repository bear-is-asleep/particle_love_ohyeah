import numpy as np
import pandas as pd
from time import time

class Grid:
    def __init__(self, xmin,xmax,ymin,ymax,zmin,zmax, divisions):
        #Define grid boundaries
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.divisions = divisions
        
        #Define grid properties
        self.attributes = ['id','x','y','z']
        
        #Initialize grid
        self.init_grid()
        self.grid_min = np.array([self.xmin,self.ymin,self.zmin])
        
    def init_grid(self):
        
        #Split the grid into n_grid_splits along each axis
        xdivisions = np.linspace(self.xmin,self.xmax,self.divisions+2)
        self.dx = xdivisions[1]-xdivisions[0]
        ydivisions = np.linspace(self.ymin,self.ymax,self.divisions+2)
        self.dy = ydivisions[1]-ydivisions[0]
        #only one z division for now
        zdivisions = np.linspace(self.zmin,self.zmax,2)
        self.dz = zdivisions[1]-zdivisions[0]
        
        self.grid = np.zeros((self.divisions+1,self.divisions+1,1,len(self.attributes)))
        
        iden = 0
        for i in range(len(xdivisions)-1):
            for j in range(len(ydivisions)-1):
                self.grid[i,j,0,0] = iden
                self.grid[i,j,0,1] = xdivisions[i] + self.dx/2
                self.grid[i,j,0,2] = ydivisions[j] + self.dy/2
                self.grid[i,j,0,3] = zdivisions[0] + self.dz/2
                iden += 1
        self.meshgrid = np.meshgrid(xdivisions,ydivisions,zdivisions,indexing='ij')
        self.ids = self.grid[:,:,:,0].flatten()
    def get_id(self, position):
        #Grid search for which grid the particle is in
        relative_pos = (position - self.grid_min) / np.array([self.dx,self.dy,self.dz])
        idx, idy, idz = np.floor(relative_pos).astype(int)

        # Check bounds to ensure the indices are within the grid size
        if not (0 <= idx < self.divisions+1 and 0 <= idy < self.divisions+1 and 0 <= idz < 1):
            #put it in the closest grid
            idx = np.clip(idx,0,self.divisions)
            idy = np.clip(idy,0,self.divisions)
            idz = np.clip(idz,0,0)
            #raise ValueError(f'Particle at position {position} is outside the grid')
            
        #Return the grid id
        return self.grid[idx,idy,idz,0] 
    def show_lines(self,ax,**pltkwargs):
        #Show the grid lines
        xx, yy, zz = self.meshgrid
        xs = np.unique(xx.flatten())
        ys = np.unique(yy.flatten())
        for x in xs:
            ax.axvline(x,**pltkwargs)
        for y in ys:
            ax.axhline(y,**pltkwargs)
    def get_particles_for_ids(self,particles):
        #Return list the length of particles containing the list indices of particles in each grid
        masks = [None]*len(self.ids)
        grid_ids = [p.grid_id for p in particles]
        for i,grid_id in enumerate(self.ids):
            masks[i] = grid_ids == grid_id
        return masks
        
        
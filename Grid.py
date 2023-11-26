import numpy as np
import pandas as pd
from time import time
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Grid:
    def __init__(self, xmin,xmax,ymin,ymax,zmin,zmax, divisions, color='white'
                 ,attributes=['id','x','y','z']):
        #Define grid boundaries
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.divisions = divisions
        
        #Define grid properties
        self.attributes = attributes
        
        #Initialize grid
        self.init_grid()
        self.grid_min = np.array([self.xmin,self.ymin,self.zmin])
        
    def init_grid(self):
        #Split the grid into n_grid_splits along each axis
        xdivisions = np.linspace(self.xmin,self.xmax,self.divisions+2)
        self.dx = xdivisions[1]-xdivisions[0]
        ydivisions = np.linspace(self.ymin,self.ymax,self.divisions+2)
        self.dy = ydivisions[1]-ydivisions[0]
        zdivisions = np.linspace(self.zmin,self.zmax,self.divisions+2)
        self.dz = zdivisions[1]-zdivisions[0]
        self.grid = np.zeros((self.divisions+2,self.divisions+2,self.divisions+2,len(self.attributes)+1))
        
        iden = 0
        for i in range(len(xdivisions)):
            for j in range(len(ydivisions)):
                for k in range(len(zdivisions)):
                    self.grid[i,j,k,0] = iden
                    self.grid[i,j,k,1] = xdivisions[i]
                    self.grid[i,j,k,2] = ydivisions[j]
                    self.grid[i,j,k,3] = zdivisions[k]
                    for l in range(4,len(self.attributes)+1):
                        self.grid[i,j,k,l] = iden
                    iden += 1
                iden += 1
            iden+=1
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
            idz = np.clip(idz,0,self.divisions)
            #raise ValueError(f'Particle at position {position} is outside the grid')
            
        #Return the grid id
        return self.grid[idx,idy,idz,0] 
    def show_lines(self,ax,**pltkwargs):
        #Show the grid lines
        xx, yy, zz = self.meshgrid
        xs = np.unique(xx.flatten())
        ys = np.unique(yy.flatten())
        zs = np.unique(zz.flatten())
        
        # Create lines for the grid structure
        lines = []
        for x in xs:
            for z in zs:
                lines.append([(x, ys[0], z), (x, ys[-1], z)])
        for y in ys:
            for z in zs:
                lines.append([(xs[0], y, z), (xs[-1], y, z)])
        for y in ys:
            for x in xs:
                lines.append([(x, y, zs[0]), (x, y, zs[-1])])
        line_collection = Line3DCollection(lines,alpha=1/(5*self.divisions+1),**pltkwargs)

        ax.add_collection3d(line_collection)
    def get_particles_for_ids(self,particles):
        #Return list the length of particles containing the list indices of particles in each grid
        masks = [None]*len(self.ids)
        grid_ids = [p.grid_id for p in particles]
        for i,grid_id in enumerate(self.ids):
            masks[i] = grid_ids == grid_id
        return masks
        
        
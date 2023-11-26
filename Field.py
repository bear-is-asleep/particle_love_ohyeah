import torch
import numpy as np
from Grid import Grid
from globals.maps import *
from physics import fields

class Field(Grid):
    def __init__(self, xmin,xmax,ymin,ymax,zmin,zmax, divisions
                 ,attributes=['id','x','y','z','F'],dynamic=False):
        #Grid initialization
        super().__init__(xmin,xmax,ymin,ymax,zmin,zmax, divisions)  
        self.dynamic = dynamic      
class ScalarField(Field):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes=['id', 'x', 'y', 'z', 'F'], dynamic=False):
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes, dynamic)
        self.field = self.grid[:,:,:,4]
    def __str__(self):
        return f'ScalarField: {self.attributes}'
    def show_field(self, ax, **kwargs):
        #Extract values
        x = self.grid[:,:,:,1].flatten()
        y = self.grid[:,:,:,2].flatten()
        z = self.grid[:,:,:,3].flatten()
        F = self.grid[:,:,:,4].flatten()
        
        # Plot the scalar field
        sc = ax.scatter(x,y,z,c=F,s=100/(self.divisions+1),**kwargs)
        return sc

class Gravity(ScalarField):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes=['id', 'x', 'y', 'z', 'g'], dynamic=False):
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes, dynamic)
    def update(self,particle_graph,G=1):
        #Set all field values to zero
        self.grid[:,:,:,4] = 0
        
        distances = particle_graph.edge_attr[:,:3]
        separations = particle_graph.edge_attr[:,3]
        masses = particle_graph.node_attr[:,M_IND]
        edge_mask = particle_graph.edge_mask
        print('distances',distances)
        print('separations',separations)
        print('masses',masses)
        print('edge_mask',edge_mask)
        
        field = fields.compute_gravity_field(distances,separations,masses,edge_mask,G=G)
        print('field',field)
        
        
        
        
        
        
    
        
        
        
        
        

        
     
class VectorField(Field):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes=['id', 'x', 'y', 'z', 'F'], dynamic=False):
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, divisions, attributes, dynamic)
        self.field = self.grid[:,:,:,4:7]
import numpy as np
import numba as nb
import sys
import pyopencl as cl
from utils import gpu_helpers as gpu


@nb.jit(nopython=True)
def calc_force_parts_cpu(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1):
    force = np.zeros(3) #3 vector
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue 
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i])) #Prevent division by zero
        rhat = displacement_vector[i]/r
        print('rhat:',rhat)
        
        #Forces
        force += rhat*G*mass*other_masses[i]/r**2 #gravity
        force -= rhat*k*charge*other_charges[i]/r**2 #electric
        force += rhat*K*spin*other_spins[i]*r**2 #spring
    return force #3 vector

#Code for gpu functions
force_code = """
__kernel void calc_force_parts(__global float3* displacement_vector, 
                            float mass, __global float* other_masses,
                            float charge, __global float* other_charges,
                            float spin, __global float* other_spins,
                            __global float* separation_vector,
                            const unsigned int num_particles,
                            float G, float K, float k, __global float3* force_out) {
int i = get_global_id(0);  // Get the ID of the current work item.

printf("displacement_vector size: %d\\n", sizeof(displacement_vector));

if (i >= num_particles) {
    return; // Avoid accessing out of bounds
}   

float3 force = (float3)(0.0f, 0.0f, 0.0f); // Initialize force to zero

// Loop over all particles to calculate force
printf("num_particles: %d\\n", num_particles);
for (int j = 0; j < num_particles; j++) {
    float3 disp = displacement_vector[j];
    printf("disp: %f %f %f\\n", disp.x, disp.y, disp.z);
    printf("other_masses: %f\\n", other_masses[j]);
    
    if (disp.x == 0.0f && disp.y == 0.0f && disp.z == 0.0f) continue; // Skip if no displacement (it's the same particle)
    float r = separation_vector[j];
    float r_norm = length(disp); // Calculate the magnitude of displacement

    if (r_norm < r) {
        r_norm = r; // Maximum separation is the radius of particles
    }

    float3 rhat = disp / r_norm; // Normalize the displacement vector

    // Calculate forces based on the provided equations
    //printf("rhat: %f %f %f\\n", rhat.x, rhat.y, rhat.z);
    
    force += rhat * G * mass * other_masses[j] / (r_norm * r_norm);
    force -= rhat * k * charge * other_charges[j] / (r_norm * r_norm);
    force += rhat * K * spin * other_spins[j] / (r_norm * r_norm);
}


// Write the calculated force to the output buffer
force_out[i] = force;
    
}
"""

def calc_force_parts_gpu(ctx,queue,prg,displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1):
    assert ctx is not None and queue is not None and prg is not None, f'ctx{ctx}, queue{queue}, and prg{prg} must be provided'
    # Prepare data
    displacement_vector = np.array(displacement_vector, dtype=np.float32) 
    other_masses = np.array(other_masses, dtype=np.float32)
    other_charges = np.array(other_charges, dtype=np.float32)
    other_spins = np.array(other_spins, dtype=np.float32)
    separation_vector = np.array(separation_vector, dtype=np.float32)
    num_particles = np.int32(len(other_spins.copy()))
    force_vector = np.empty((num_particles,3), dtype=np.float32)
    # if num_particles <=1:
    #     print('--Warning: num_particles <=1, verify this??')
    # print('displacement_vector:',displacement_vector.copy())
    
    #print shapes and sizes of arrays
    print('displacement_vector:',displacement_vector.shape,displacement_vector.nbytes)
    # print('other_masses:',other_masses.shape,other_masses.nbytes)
    # print('other_charges:',other_charges.shape,other_charges.nbytes)
    # print('other_spins:',other_spins.shape,other_spins.nbytes)
    # print('separation_vector:',separation_vector.shape,separation_vector.nbytes)
    # print('num_particles:',num_particles)
    # print('force_vector:',force_vector.shape,force_vector.nbytes)
    

    # Allocate memory for input and output data on the device.
    displacement_vector_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=displacement_vector.nbytes,hostbuf=displacement_vector)
    other_masses_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=other_masses.nbytes,hostbuf=other_masses)
    other_charges_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=other_charges.nbytes,hostbuf=other_charges)
    other_spins_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=other_spins.nbytes,hostbuf=other_spins)
    separation_vector_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=separation_vector.nbytes,hostbuf=separation_vector)

    # Create output buffer
    force_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=force_vector.nbytes)  # Allocate space for output.

    # Set kernel arguments
    global_work_size = (num_particles,)

    prg.calc_force_parts(queue, global_work_size, None, 
                     displacement_vector_buf,
                     np.float32(mass), 
                     other_masses_buf, 
                     np.float32(charge),
                     other_charges_buf, 
                     np.float32(spin), 
                     other_spins_buf,
                     separation_vector_buf, 
                     np.int32(num_particles),
                     np.float32(G), np.float32(K),
                     np.float32(k), force_buf)

    #Read the output buffer back into a Python variable.
    queue.finish()
    partial_force_vectors = np.empty((num_particles, 3), dtype=np.float32)
    copy_event = cl.enqueue_copy(queue, partial_force_vectors, force_buf)

    force = np.sum(partial_force_vectors,axis=0)
    
    #Release memory
    # displacement_vector_buf.release()
    # other_masses_buf.release()
    # other_charges_buf.release()
    # other_spins_buf.release()
    # separation_vector_buf.release()
    # force_buf.release()

    return force #3 vector

def compare_force_parts(ctx,queue,prg,displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1):
    cpu = calc_force_parts_cpu(displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1)
    gpu = calc_force_parts_gpu(ctx,queue,prg,displacement_vector,mass,other_masses,charge,other_charges,spin,other_spins,separation_vector
                        ,G=1,K=1,k=1)
    print('CPU:',cpu)
    print('GPU:',gpu)
    print('Difference:',cpu-gpu)
    print('Mag dif:',np.linalg.norm(cpu-gpu))
    print('norm to cpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu))
    print('norm to gpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(gpu))
    print('norm to gpu+cpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu+gpu))
    print('norm to cpu-gpu:',np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu-gpu))

@nb.jit(nopython=True)
def calc_electric_parts(displacement_vector,charge,other_charges,separation_vector,k=1):
    force = np.zeros(3)
    for i in range(len(displacement_vector)):
        if np.all(displacement_vector[i] == 0): continue
        r = max(separation_vector[i],np.linalg.norm(displacement_vector[i]))
        rhat = displacement_vector[i]/r
        force += rhat*k*charge*other_charges[i]/r**2
    return force

@nb.jit(nopython=True)
def calc_collision_parts(displacement_vector,mass,other_masses,velocity,other_velocites,radius,other_radii):
    #TODO: Implement this
    pass
            
import pyopencl as cl

def initialize_gpu_function(kernel_code):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    return ctx, queue, prg
    
import subprocess
from matplotlib import animation

def is_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
    
def save_animation(ani, filename, fps=24, bitrate=1800):
    if is_ffmpeg_installed() and filename.endswith('.mp4'):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=bitrate)
        ani.save(filename, writer=writer)
    elif filename.endswith('.mp4'):
        print('ffmpeg not installed. Defaulting to gif')
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps, bitrate=bitrate)
        ani.save(filename, writer='pillow')
    elif filename.endswith('.gif'):
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps, bitrate=bitrate)
        ani.save(filename, writer='pillow')
    elif filename.endswith('.html'):
        Writer = animation.writers['html']
        writer = Writer(fps=fps, bitrate=bitrate)
        ani.save(filename, writer='html')
    else:
        raise ValueError('Invalid file extension')
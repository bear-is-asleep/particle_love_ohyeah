from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.widgets import Slider

def map_value_to_color(value, min_val, max_val, cmap='viridis',return_hex=True):
    # Create a colormap
    cmap = plt.get_cmap(cmap)  # replace 'viridis' with your colormap
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    normalized_value = norm(value)
    rgb_color = cmap(normalized_value)
    if return_hex:
        return mcolors.rgb2hex(rgb_color[:3])
    return rgb_color
  
def set_style(ax, facecolor='black', axis_off=True):
    ax.set_facecolor(facecolor)
    if axis_off:
        ax.axis('off')
        
def add_slider(self, ax, label, valmin, valmax, valinit, valfmt):
    slider = Slider(ax, label, valmin, valmax, valinit=valinit, valfmt=valfmt)
    slider.on_changed(self.update_parameters)
    return slider
        
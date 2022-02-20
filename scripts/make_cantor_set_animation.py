import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from PIL import Image
from turing.ss.simulator import encoding_function


def show_encoding_in_cantor_set_grid(s: str):
    face_color = 'dimgrey'
    axis_color = 'white'
    grid_color = 'white'
    cantor_set_color = 'darkgrey'
    active_encoding_color = 'cyan'
    active_text_color = 'cyan'

    fig, ax = plt.subplots(1, 1)   
       
    strings=['']
    u, l = 0, -1
    for level in range(5):  
        strings = [s+'0' for s in strings] + [s+'1' for s in strings]    
        for a in strings:
            x = encoding_function(a, base=4, p=1/2)
            plt.plot([x, x], [u, l], color=cantor_set_color, linewidth=2)
        u, l = l, l - (u - l)

    u, l = -len(s)+1, -len(s)
    x = encoding_function(s, base=4, p=1/2)
    plt.plot([x, x], [u, l], color=active_encoding_color, linewidth=2)
    plt.text(x, .5*(u+l), f' \"{s}\"', color=active_text_color, weight='bold')
    
    # x axis
    ax.set_xlabel('encoding')
    ax.spines['top'].set_position('zero')
    ax.spines['top'].set_color(axis_color)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    grid = .5**4
    rug = torch.arange(0, 1.1, grid)
    plt.xticks(rug.tolist(), rotation=60, color=axis_color)
    ax.xaxis.label.set_color(axis_color)    
    ax.xaxis.set_label_position('top') 
    ax.tick_params(axis='x', colors=axis_color)    
    plt.xlim([0, 1])

    # y axis
    ax.set_ylabel('string length')
    ax.spines['left'].set_color(axis_color)    
    ax.spines['right'].set_visible(False)
    plt.yticks([-1, -2, -3, -4, -5], color=axis_color)    
    ax.tick_params(axis='y', colors=axis_color)
    ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.yaxis.label.set_color(axis_color)    
    plt.ylim([-5, 0])

    # down arrow
    ax.plot(0, -5, ls="", marker="v", ms=10, color=axis_color,
        transform=ax.get_yaxis_transform(), clip_on=False)
    
    # background    
    fig.set_facecolor(face_color)
    ax.set_facecolor(face_color)
    ax.grid(True, linewidth=0.2, color=grid_color)
    return fig, ax


ims = list()
counter = 0
fig_names = []
for s in ["0", "10", "110", "0110", "10110", "0", "00", "000", "0000", "00000"]:
    fig, ax = show_encoding_in_cantor_set_grid(s)
    name = f'show_encoding_in_cantor_set_grid_{counter}_{s}.png'
    fig.savefig(name, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig_names.append(name)


imgs = (Image.open(f) for f in fig_names)
img = next(imgs)  # extract first image from iterator
img.save(fp='show_encoding_in_cantor_set_grid.gif', format='GIF', append_images=imgs,
         save_all=True, duration=1000, loop=0)


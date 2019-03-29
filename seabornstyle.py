## seaborn style

import seaborn as sns; sns.set()

mowbray_colors = ['#01597f', '#0b87a1', '#80bfb7', '#d3ebd5']

def set_style():
    sns.set_context("talk");
    sns.set(font='serif');
    sns.set_style("darkgrid", {
        'font.family' : 'serif',
        'font.sans-serif' : ["DejaVu Sans", "Arial"],
        'font.serif' : ["Palatino", "serif"],
        });
    sns.set_palette(mowbray_colors)
    
    
set_style()

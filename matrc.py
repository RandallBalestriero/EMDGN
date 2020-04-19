from matplotlib import rc, rcParams
from cycler import cycler


rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True, hinting_factor=8)
rc('xtick', labelsize='xx-large', direction='out')
rc('ytick', labelsize='xx-large', direction='out')
rc('figure', figsize=(5, 5))
rc('axes', grid=True, titlesize='x-large', labelsize='large',
        prop_cycle=cycler(color=['#348ABD', '#A60628', '#7A68A6', '#467821', 
            '#D55E00',  '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']))
rc('lines', linewidth=1.7, antialiased=True)
rc('grid', alpha=0.3)
rcParams['backend']='Agg'
rc('image', cmap='plasma', origin='lower', aspect='auto', 
        interpolation='bicubic')
rc('legend', fancybox=True, fontsize='xx-large', handletextpad=0.2,
        handlelength=0.7)
rc('figure', autolayout=True)

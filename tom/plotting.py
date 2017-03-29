
import matplotlib

# Qt4Agg was seg faulting for some unclear reasons with savefig
# stopped happening once i got rid of the dpi argument
'''
# can't use
#backend = 'GTKAgg'
backend = 'Qt4Agg'

# segfaulted so far...
#backend = 'TkAgg'
#backend = 'Qt5Agg'

# checks the backend is in the *hardcoded* list of supported backends
assert matplotlib.rcsetup.validate_backend(backend), 'invalid backend'
print('USING ' + backend)
matplotlib.use(backend)
'''

import matplotlib.pyplot as plt
#print('actually using backend ' + plt.get_backend())

import seaborn as sns
import numpy as np
import pandas as pd


def plot_image_dict(image_dict):
    key2sbplt = dict()
    sbplt2key = dict()

    for i, k in enumerate(sorted(data_dict.keys()), start=1):
        key2sbplt[k] = i
        sbplt2key[i] = k

    #rows = min(len(data_dict.keys()), 2)
    rows = 4
    cols = int(np.ceil(len(data_dict.keys()) / rows))

    fig, axarr = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)

    '''
    # to center colormap quickly. hacky.
    vmax = max_value(data_dict) * 0.3
    vmin = -vmax
    '''
    vmax = max_value(data_dict)
    vmin = min_value(data_dict)

    # turn off any subplots we wont actually use
    for i in range(rows):
        for j in range(cols):
            curr = i*cols + j

            if cols > 1:
                ax = axarr[int(np.floor(curr / cols))][curr % cols]
            elif type(axarr) == np.ndarray:
                ax = axarr[curr]
            else:
                ax = axarr
            sbplt = curr + 1

            if sbplt in sbplt2key:
                k = sbplt2key[sbplt]

                try:
                    img = ax.imshow(data_dict[k], vmin=vmin, vmax=vmax)
                except TypeError:
                    # sometimes imshow will give a TypeError with a message:
                    raise TypeError("Invalid dimensions for image data: " + \
                            str(data_dict[k].shape))

                img.set_cmap(cmap)
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])

                # to get rid of grey margins
                # how is this different from 'box' and 'datalim'? not clipping is it?
                ax.set_adjustable('box-forced')
                ax.title.set_text(k)

            else:
                if cols == 1:
                    axarr[i].axis('off')
                else:
                    axarr[i,j].axis('off')

    if image:
        # TODO units correct? test
        # latex percent sign?
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = fig.colorbar(img, cax=cax)
        cb.ax.set_title(r'$\frac{\Delta{}F}{F}$')

    # TODO why was only wspace working when both were 0? it might still be only one working
    fig.subplots_adjust(wspace=0.005, hspace=0.2)
    return fig


def plot(data, title=None, cmap=None, window_title=None, save=False):
    sns.set_style('dark')

    if type(data) is dict:
        fig = plot_image_dict(data)
    else:
        fig = plt.figure()
        plt.imshow(data)

    # will these still work if it plt isn't a subplots object?
    if not title is None:
        plt.suptitle(title)

    if not window_title is None:
        fig.canvas.set_window_title(window_title)

    if save:
        prefix = './figures/'
        fname = prefix + title.replace(' ', '').replace(',', '').replace('odorpanel', '_o').\
                replace('=', '') + '.eps'

        print('saving fig to ' + fname + ' ... ', end='')

        # dpi=9600 caused it to crash, so now i can just control whether text of 
        # neighboring subplots overlaps by changing the figure size
        fig.set_size_inches(12, 12)
        fig.savefig(fname)

        print('done')
    return fig


def hist_image(img, title=''):
    """
    For sanity checking some image processing operations
    """

    fig = plt.figure()
    n, bins, patches = plt.hist(img.flatten(), 50, normed=1)

    if not title == '':
        plt.title(title)
    else:
        plt.title('Image histogram, dtype=' + str(img.dtype))

    plt.xlabel('Pixel intensity')
    plt.ylabel('Probability')
    return fig


def summarize_fly(fly_df):
    # TODO handle manual rois

    glomeruli = set(fly_df.columns)
    session = list(fly_df.index.get_level_values('fly_id').unique())[0]

    #glomeruli.remove('block')
    #print(glomeruli)

    # TODO just make sublots each block?
    for glom in glomeruli:
        ##############################################################################
        # plot this session's individual traces, extracted from the ROIs
        ##############################################################################

        print(glom)
        # TODO assert somehow that a block either has the glomerulus in all frames / odors
        # or doesnt?
        # get the entries (?) that have data for that glomerulus
        glom_df = fly_df[glom][pd.notnull(fly_df[glom])]

        if sum(glom_df.shape) == 0:
            continue

        #containing_blocks = fly_df[pd.notnull(fly_df[glom])].reset_index()['block']

        # TODO grid? units of seconds on x-axis. patch in stimulus presentation.
        # TODO check for onsets before we would expect them
        df = glom_df.reset_index()

        # palette=sns.color_palette("Blues_d"))
        # plot individual traces
        g = sns.FacetGrid(df, hue='trial', col='odor', col_wrap=5)
        g = g.map(plt.plot, 'frame', glom, marker='.')

        g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
        title = session + ' ' + glom
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9)

        '''
        # get set of blocks occurring with current odor (column name, the arg to lambda)
        f = lambda x: x + ' ' + str(list(filter(lambda e: type(e) is int, \
                set(containing_blocks.where(df['odor'] == x).unique()))))
        g.set_titles(col_func=f)
        '''


        ##############################################################################
        # plot this session's mean traces, extracted from the ROIs
        ##############################################################################

        # plot means w/ SEM errorbars
        df = glom_df.reset_index()
        df[glom] = df[glom].apply(pd.to_numeric)

        #grouped = glom_df.groupby(level=['odor', 'trial'])
        grouped = df.groupby(['odor', 'frame'])
        means = grouped.mean()
        means['sem'] = grouped[glom].sem()
        mdf = means.reset_index()
        g = sns.FacetGrid(mdf, col='odor', col_wrap=5)
        g = g.map(plt.errorbar, 'frame', glom, 'sem')

        g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
        #title = 'Mean w/ SEM for ' + glom + ' from blocks ' + str(containing_blocks.unique())
        title = session + ' mean and SEM for ' + glom
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9)


def summarize_flies(projections, rois, df, save_to=None):
    # we don't care which condition they came from
    # i feel like i should be able to say 'fly_id' instead of level=1, but i tried...
    grouped = df.groupby(level=1).apply(summarize_fly)

    # TODO
    '''
    for session, image_dict in projections.items():
        plot(image_dict, title='', window_title='', save=True)

    for session, image_dict in rois.items():
        plot(image_dict, title='', window_title='', save=True)
    '''


def summarize_experiment(df, save_to=None):
    pass
    

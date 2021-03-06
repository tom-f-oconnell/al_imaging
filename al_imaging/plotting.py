
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
from os.path import split, join, exists
import os

import matplotlib.pyplot as plt
#print('actually using backend ' + plt.get_backend())

import seaborn as sns
import numpy as np
import pandas as pd

# TODO factor this back in to analysis?
import cv2
import random

from . import odors

def reduce(v, c, data):
    """
    data: np.ndarray or tuple / list of them
    v: function that takes elements of data to some scalar (or something comparable)
    c: function that compares two outputs of v(dn), and returns True or False
       (condition under which to update state)
    """

    if type(data) is dict:
        mapped = tuple(map(lambda x: reduce(v, c, x), data.values()))
        return reduce(lambda x: x, c, mapped)

    elif type(data) is list or type(data) is tuple:
        m = data[0]

        for d in data:
            val = v(d)

            if c(m,val):
                m = val

        return m

    else:   
        return v(data)


def max_value(data):
    # this is highlighted, but it is not actually a keyword in Python 3
    return reduce(np.max, lambda curr, v: curr < v, data)


def min_value(data):
    return reduce(np.min, lambda curr, v: curr > v, data)


def contour2img(contour, shape):
    contour_img = np.ones((*shape, 3)).astype(np.uint8)

    # args: destination, contours, contour id (neg -> draw all), color, thickness
    cv2.drawContours(contour_img, contour, -1, (0,0,255), 3)

    x, y, w, h = cv2.boundingRect(contour)
    # args: destination image, point 1, point 2 (other corner), color, thickness
    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0,255,0), 5)

    return contour_img


def plot_image_dict(image_dict, title_prefix='', cmap='viridis'):
    key2sbplt = dict()
    sbplt2key = dict()

    for i, k in enumerate(sorted(image_dict.keys()), start=1):
        key2sbplt[k] = i
        sbplt2key[i] = k

    #rows = min(len(image_dict.keys()), 2)
    #rows = 4
    #cols = int(np.ceil(len(image_dict.keys()) / rows))
    rows = 2
    cols = 4

    fig, axarr = plt.subplots(rows, cols, sharex=True, sharey=True)
    '''
    # to center colormap quickly. hacky.
    vmax = max_value(image_dict) * 0.3
    vmin = -vmax
    '''
    vmax = max_value(image_dict)
    vmin = min_value(image_dict)

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
                    img = ax.imshow(image_dict[k], vmin=vmin, vmax=vmax)
                except TypeError:
                    # sometimes imshow will give a TypeError with a message:
                    raise TypeError("Invalid dimensions for image data: " + \
                            str(image_dict[k].shape))

                img.set_cmap(cmap)
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])

                # to get rid of grey margins
                # how is this different from 'box' and 'datalim'? not clipping is it?
                ax.set_adjustable('box-forced')
                # TODO just convert to strings ahead of time
                if type(k) is frozenset:
                    #k = str(set(odors.pair2str(e) for e in k))
                    k = '{' + ',\n'.join([odors.pair2str(e) for e in k]) + '}'

                ax.title.set_text(k)

            else:
                if cols == 1:
                    axarr[i].axis('off')
                else:
                    axarr[i,j].axis('off')

    # TODO units correct? test
    # latex percent sign?
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = fig.colorbar(img, cax=cax)
    cb.ax.set_title(r'$\frac{\Delta{}F}{F}$')

    # TODO why was only wspace working when both were 0? it might still be only one working
    fig.subplots_adjust(wspace=0.005, hspace=0.2)
    return fig


def plot(data, title_prefix=None, title=None, cmap=None, window_title=None, \
        save_to=None, file_prefix=''):
    sns.set_style('dark')

    if type(data) is dict:
        if len(data) == 0:
            return
        fig = plot_image_dict(data, title_prefix=title_prefix)
    else:
        fig = plt.figure()
        plt.imshow(data)

    # will these still work if it plt isn't a subplots object?
    if not title is None:
        print(title_prefix)
        plt.suptitle(title_prefix)

    if not window_title is None:
        fig.canvas.set_window_title(window_title)

    if save_to is not None:
        prefix = save_to
        # TODO is this still what i want? am i overwriting stuff?
        fname = join(prefix, file_prefix + title_prefix.replace(' ', '').replace(',', '').\
                replace('odorpanel', '_o').replace('=', '') + '.png')
        # used to be .eps

        print('saving fig to ' + fname + ' ... ', end='')

        # dpi=9600 caused it to crash, so now i can just control whether text of 
        # neighboring subplots overlaps by changing the figure size
        #side = 18
        side = 12
        fig.set_size_inches(side, side)
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
    glomeruli = set(fly_df.columns)
    session = list(fly_df.index.get_level_values('fly_id').unique())[0]

    # TODO just make sublots each block?
    for glom in glomeruli:
        ##############################################################################
        # plot this session's individual traces, extracted from the ROIs
        ##############################################################################

        # TODO assert somehow that a block either has the glomerulus in all frames / odors
        # or doesnt?
        # get the entries (?) that have data for that glomerulus
        glom_df = fly_df[glom][pd.notnull(fly_df[glom])]

        # check how many times max frame # divides size?

        # not sure why this didn't happen on most recent run...
        if sum(glom_df.shape) == 0:
            continue

        #containing_blocks = fly_df[pd.notnull(fly_df[glom])].reset_index()['block']

        # TODO grid? units of seconds on x-axis. patch in stimulus presentation.
        # TODO check for onsets before we would expect them
        df = glom_df.reset_index()

        # palette=sns.color_palette("Blues_d"))
        # plot individual traces
        g = sns.FacetGrid(df, hue='trial', col='odor', col_wrap=5)
        g = g.map(plt.plot, 'timepoint', glom, marker='.')

        g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
        title = session + ' ' + glom
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9)

        # TODO fix titles. including seconds not frames.
        f = lambda x: '{' + ',\n'.join([odors.pair2str(e) for e in x]) + '}'
        g.set_titles(col_func=f)


def summarize_flies(projections, rois, df, save_to=None):
    if save_to is not None and not exists(save_to):
        os.makedirs(save_to)

    # TODO include session suffix in actual session. will need to change in analysis.
    #for session, image_dict in projections.items():
    #    plot(image_dict, title_prefix=split(session)[-1] + ' ', save_to=save_to, file_prefix='max')

    # commented because the ijrois are already masks.
    # TODO need a solution that works for both!!!
    # turn contours described by points around perimeter to images summarizing them
    #rois = {k: {g: contour2img(i, list(projections[k].items())[0][1].shape) for g, i in v.items()} \
    #        for k, v in rois.items()}
    #for session, image_dict in rois.items():
    #    print('session', session)
    #    plot(image_dict, title_prefix=split(session)[-1] + ' ', save_to=save_to, file_prefix='roi')

    # we don't care which condition they came from
    # i feel like i should be able to say 'fly_id' instead of level=1, but i tried...
    grouped = df.groupby(level=1).apply(summarize_fly)
    # TODO save these figures


def summarize_experiment(df, save_to=None):
    if save_to is not None and not exists(save_to):
        os.makedirs(save_to)

    # TODO put these in a config file or something...
    for glom in ('vm7', 'dm6', 'dl5'):
        #glom_df = fly_df[glom][pd.notnull(fly_df[glom])]

        l = []
        # TODO make case insensitive
        for col in df.filter(regex=glom, axis=1):
            l.append(df[col])
        if len(l) == 0:
            continue

        glom_df = pd.DataFrame()
        glom_df[glom] = pd.to_numeric(pd.concat(map(lambda x: x.dropna(), l)))

        grouped = glom_df.groupby(level=['condition', 'odor', 'timepoint'])
        # TODO name other col mean rather than glom name
        to_plot = grouped.mean()
        to_plot['sem'] = grouped[glom].sem()
        mdf = to_plot.reset_index()

        g = sns.FacetGrid(mdf, col='odor', hue='condition', col_wrap=5)
        # TODO sem per each glom is what is happening, right?
        g = g.map(plt.errorbar, 'timepoint', glom, 'sem').add_legend()

        g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
        #title = 'Mean w/ SEM for ' + glom + ' from blocks ' + str(containing_blocks.unique())
        title = 'Mean and SEM for ' + glom
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9)
        f = lambda x: '{' + ',\n'.join([odors.pair2str(e) for e in x]) + '}'
        g.set_titles(col_func=f)
        

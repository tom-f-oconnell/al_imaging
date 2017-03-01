
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# bokeh too later maybe

# TODO how to handle equivalent of subtitles grouping as before?
# should I?
# TODO handle at least images and dot / line plotting
# TODO default to turn off all tickmarks and stuff for images

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

    #elif type(data) == np.ndarray or type(data) == np.float64:
    else:   
        return v(data)

    '''
    else:
        assert False, 'can only reduce over numpy array or list/tuple of them, not ' + \
                str(type(data))
    '''

def max_value(data):
    # this is highlighted, but it is not actually a keyword in Python 3
    return reduce(np.max, lambda curr, v: curr < v, data)

def min_value(data):
    return reduce(np.min, lambda curr, v: curr > v, data)

def first_array(data):
    if type(data) is dict:
        v = list(data.values())[0]
    else:
        v = data

    # asserts that the groupings are only singly nested, which seems reasonable for now
    if type(v) is list or type(v) is tuple:
        return v[0]
    
    else:
        return v

def is_image(v):
    """ assumes things not 1 dimensional are images """

    # only true when one dimension has greater than 1 indices, and all others have 1
    return not np.prod(v.shape) == np.max(v.shape)

def dict2subplots(data_dict, xs, sharex, sharey, avg, cmap, image, emin=None, emax=None):
    '''
    data_dict should be of the form:
        {key (relevant to experimental condition)-> Tx1 (or 1xT? / (T,)?) numpy (?) 1d series
                                                                       or
                                                    NxN image
                                                                       or
                                                    tuple or list containing exactly one of the above
        , key2 -> d2, ...}

    other parameters can override some matplotlib defaults

    avg only meaningful if values are tuples or lists, in which case an average for each group
    of values will also be displayed

    inherits the default kwargs of plot below
    '''

    # right now we have to pick which images we want from an image series in advance

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

    if not image:
        border = 0.10
        vmin = vmin * (1 - border)
        vmax = vmax * (1 + border)

    # TODO integrate into below
    '''
                curr = i*cols + j

                if curr >= len(subtitles):
                    break

                if cols == 1:
                    ax = axarr[i]
                else:
                    ax = axarr[i,j]

                for w in windows[curr]:
                    ax.plot(times, signal[curr][w[0]:w[1]], alpha=0.6, linewidth=0.3)

                ax.set_ylim((ymin, ymax))
                ax.fill_between(times, ymin, ymax, where=trigger_in_window.flatten() > 2.5, \
                        facecolor='black', alpha=0.1)

                curr_pins = set(pin2avg[curr].keys())
                assert len(curr_pins) == 1, 'havent implemented multiple pins + arbitrary group'

                ax.plot(times, pin2avg[curr][curr_pins.pop()], '-', c='black', alpha=0.6, \
                        linewidth=1.5, label='Mean')

                if curr == 0:
                    ax.legend()

                ax.title.set_text(subtitles[curr])
    '''

    # turn off any subplots we wont actually use
    for i in range(rows):
        for j in range(cols):
            #curr = i*cols + j + 1
            curr = i*cols + j

            # TODO make shorter?
            if cols > 1:
                ax = axarr[int(np.floor(curr / cols))][curr % cols]
            elif type(axarr) == np.ndarray:
                ax = axarr[curr]
            else:
                ax = axarr

            sbplt = curr + 1
            if sbplt in sbplt2key:
                k = sbplt2key[sbplt]

                if image:
                    # TODO:
                    # -artifacts? only reason scale is off?
                    # -make plots close together
                    # -border vs change background color?

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

                elif not xs is None:
                    ax.plot(xs, data_dict[k], alpha=0.6, linewidth=0.3)

                else:
                    # hist?
                    #assert False, 'not yet implemented'
                    # TODO fix
                    clow = emin[k]
                    chigh = emax[k]
                    ax.errorbar(np.arange(data_dict[k].shape[0]), data_dict[k], yerr=[clow, chigh])

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

def plot(data, xs=None, title=None, sharex=True, sharey=True, avg=True, cmap=None, \
        window_title=None, emin=None, emax=None, save=False):

    # check whether we are dealing with images or a 1d series
    image = is_image(first_array(data))

    if image:
        sns.set_style('dark')
    else:
        # is this the right name?
        sns.set_style('darkgrid')

    if type(data) is dict:
        # TODO why did i ever include the xs argument?
        fig = dict2subplots(data, xs, sharex, sharey, avg, cmap, image, emin, emax)
        # TODO
    else:
        fig = plt.figure()
        #plt.plot / plt.imshow

        if image:
            plt.imshow(data)

        elif not xs is None:
            # default dt?
            plt.plot(xs, data, alpha=0.6, linewidth=0.3)

        else:
            assert False, 'not yet implemented'

        # TODO if xs is None, histogram?

    # will these still work if it plt isn't a subplots object?
    if not title is None:
        plt.suptitle(title)

    if not image:
    # TODO make these defaults? if i plan to use this besides for PID / images
        # (I would rather not have to manually set the position of these, but seems like common way
        #  to do it)
        fig.text(0.5, 0.04, 'Time (frames)', ha='center')
        fig.text(0.04, 0.5, r'$\frac{\Delta{}F}{F}$ in ROI', va='center', rotation='vertical')

    if not window_title is None:
        fig.canvas.set_window_title(window_title)

    if save:
        prefix = './figures/'
        fname = prefix + title.replace(' ', '').replace(',', '').replace('odorpanel', '_o').\
                replace('=', '') + '.pdf'
        print('SAVING FIG TO ' + fname)
        plt.savefig(fname, dpi=9600)
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

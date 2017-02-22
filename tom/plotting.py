
import matplotlib.pyplot as plt
import seaborn as sns

# bokeh too later maybe

# TODO how to handle equivalent of subtitles grouping as before?
# should I?

# TODO handle at least images and dot / line plotting

# TODO default to turn off all tickmarks and stuff for images

def dict2subplots(data_dict, sharex, sharey, avg):
    '''
    data_dict should be of the form:
        {key (relevant to experimental condition)-> Tx1 (or 1xT? / (T,)?) numpy (?) 1d series
                                                                       or
                                                    TxNxN image series
                                                                       or
                                                    tuple or list containing exactly one of the above
        , key2 -> d2, ...}

    other parameters can override some matplotlib defaults

    avg only meaningful if values are tuples or lists, in which case an average for each group
    of values will also be displayed
    '''

    key2sbplt = dict()
    sbplt2key = dict()

    for i, k in enumerate(data_dict.keys(), start=1):
        key2sbplt[k] = i
        sbplt2key[i] = k

    rows = min(len(pin2odor.keys()), 2)
    cols = int(np.ceil(len(unique_pins) / rows))

    fig, axarr = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)

    # turn off any subplots we wont actually use
    for i in range(row):
        for j in range(cols):
            curr = i*cols + j + 1

            elif not pins is None and not curr in sbplt2pin:
                if cols == 1:
                    axarr[i].axis('off')
                else:
                    axarr[i,j].axis('off')

    return fig

#def fill_sbplt(axarr, key2sbplt):

def plot(data, title=None, sharex=True, sharey=True, avg=True):
    if type(data) is dict:
        fig = dict2subplots(data, titles, sharex, sharey, avg)
        # TODO
    else:
        fig = plt.plot()

    # will these still work if it plt isn't a subplots object?
    plt.suptitle(title)

    # (I would rather not have to manually set the position of these, but seems like common way
    #  to do it)
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center')
    fig.text(0.04, 0.5, 'PID output voltage', va='center', rotation='vertical')

    if not window_title is None:
        fig.canvas.set_window_title(window_title)

    return fig



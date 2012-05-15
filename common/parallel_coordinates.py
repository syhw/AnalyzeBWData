#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# NOT Copyrighted by me, see:
# http://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib

def parallel_coordinates(data_sets, style=None, x_labels=None):

    dims = len(data_sets[0])
    x    = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)

    if style is None:
        style = ['r-']*len(data_sets)

    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) / 
                min_max_range[dimension][2] 
                for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets


    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi])
        ax.set_xlim([x[i], x[i+1]])
        #ax.set_xticklabels(x_labels[i])
        # TODO BUG here with xticklabels
        print i,
        print x_labels[i]
        xtickNames = plt.setp(ax, xticklabels=x_labels[i:])
        plt.setp(xtickNames, rotation=45, fontsize=8)
    # /TODO BUG here with xticklabels

    # Set the x axis ticks 
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in xrange(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)


    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in xrange(ticks)]
    axx.set_yticklabels(labels)

    # Stack the subplots 
    plt.subplots_adjust(wspace=0)

    return plt


if __name__ == '__main__':
    import random
    base  = [0,   0,  5,   5,  0]
    scale = [1.5, 2., 1.0, 2., 2.]
    data = [[base[x] + random.uniform(0., 1.)*scale[x]
            for x in xrange(5)] for y in xrange(30)]
    colors = ['r'] * 30

    base  = [3,   6,  0,   1,  3]
    scale = [1.5, 2., 2.5, 2., 2.]
    data.extend([[base[x] + random.uniform(0., 1.)*scale[x]
                 for x in xrange(5)] for y in xrange(30)])
    colors.extend(['b'] * 30)

    parallel_coordinates(data, style=colors).savefig("parallel_coordinates.png")

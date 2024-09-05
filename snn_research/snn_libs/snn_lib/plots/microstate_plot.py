def plot_maps(maps, info, show=True):
    import mne
    import matplotlib.pyplot as plt
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_maps, n_channels)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    """
    assert len(maps) != 1, 'Only one map found, cannot plot'
    fig, axes = plt.subplots(1, len(maps), figsize=(2 * len(maps), 2))
    for i, (ax, map) in enumerate(zip(axes, maps)):
        mne.viz.plot_topomap(map, info, axes=ax, show=False)
        ax.set_title('Microstate %d' % (i+1))
    plt.tight_layout()
    if show:
        plt.show()
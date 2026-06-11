# From https://github.com/jonathanoesterle/data_plots/tree/master
import os
from collections import OrderedDict

import glasbey
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import ImageOps

# from imblearn.under_sampling import RandomUnderSampler

# Size of figures
_FIG_WIDTHS = dict(
    col=3.37,  # Single column in bioRxiv template
    full=7,  # Full width in bioRxiv template
    slide_col=5.67,  # Half powerpoint slide
    slide_full=11.5,  # Full powerpoint slide
    poster=14.46,  # Full poster width
)


def set_rc_params(kind, notebook_dpi=None):
    # Arial ttf file sould be in the work dir
    font_dir = os.path.join(os.environ['WORK'], '.fonts/mscorefonts')
    for fname in os.listdir(font_dir):
        if fname.lower().endswith('.ttf'):
            fm.fontManager.addfont(os.path.join(font_dir, fname))
    sns.set_context('paper')
    sns.set_style('ticks')
    plt.style.use(os.path.join(os.getcwd(), 'mplstyles', f'{kind}.mplstyle'))
    if notebook_dpi is not None:
        plt.rcParams['figure.dpi'] = notebook_dpi


def set_figsize(fig, width, height_ratio=None):
    if isinstance(width, str):
        if width not in _FIG_WIDTHS:
            raise NotImplementedError(f'Unknown width `{width}`')
        else:
            width = _FIG_WIDTHS[width]

    fig.set_figwidth(width)

    if height_ratio is not None:
        fig.set_figheight(width * height_ratio)


def iterate_axes(axs):
    """Make axes iterable, independent of type.
    axs (list of matplotlib axes or matplotlib axis) : Axes to apply function to.
    """

    if isinstance(axs, list):
        return axs
    elif isinstance(axs, np.ndarray):
        return axs.flatten()
    else:
        return [axs]


def grid(ax, axis='both', major=True, minor=False, **kwargs):
    """Make grid on axis"""
    for axi in iterate_axes(ax):
        if major:
            axi.grid(
                True,
                axis=axis,
                which='major',
                alpha=0.3,
                c='k',
                lw=plt.rcParams['ytick.major.width'],
                zorder=-10000,
                **kwargs,
            )
        if minor:
            axi.grid(
                True,
                axis=axis,
                which='minor',
                alpha=0.3,
                c='gray',
                lw=plt.rcParams['ytick.minor.width'],
                zorder=-20000,
                **kwargs,
            )


def dettach_axes(ax, coord=3):
    ax.spines['left'].set_position(('outward', coord))
    ax.spines['bottom'].set_position(('outward', coord))
    ax.tick_params(direction='out')


def tight_layout(h_pad=1, w_pad=1, rect=(0, 0, 1, 1), pad=None):
    """Like `tight_layout` with different default"""
    plt.tight_layout(
        h_pad=h_pad, w_pad=w_pad, pad=pad or 2.0 / plt.rcParams['font.size'], rect=rect
    )


def set_labs(
    axs,
    xlabs=None,
    ylabs=None,
    titles=None,
    panel_nums=None,
    panel_num_space=0,
    panel_num_va='bottom',
    panel_num_pad=0,
    panel_num_y=None,
):
    """Set labels and titles for all given axes.
    Parameters:

    axs : array or list of matplotlib axes.
        Axes to apply function to.

    xlabs, ylabs, titles : str, list of str, or None
        Labels/Titles.
        If single str, will be same for all axes.
        Otherwise, should have same length as axes.

    """

    for i, ax in enumerate(iterate_axes(axs)):
        if xlabs is not None:
            if isinstance(xlabs, str):
                xlab = xlabs
            else:
                xlab = xlabs[i]
            ax.set_xlabel(xlab)

        if ylabs is not None:
            if isinstance(ylabs, str):
                ylab = ylabs
            else:
                ylab = ylabs[i]
            ax.set_ylabel(ylab)

        if titles is not None:
            if isinstance(titles, str):
                title = titles
            else:
                title = titles[i]
            ax.set_title(title)

        if panel_nums is not None:
            if panel_nums == 'auto':
                panel_num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
            elif isinstance(panel_nums, str):
                panel_num = panel_nums
            else:
                panel_num = panel_nums[i]
            ax.set_title(
                panel_num + panel_num_space * ' ',
                loc='left',
                fontweight='bold',
                ha='right',
                va=panel_num_va,
                pad=panel_num_pad,
                y=panel_num_y,
            )


def map_range(value, inMin=0, inMax=20e3, outMin=3, outMax=10, reverse=True):
    # From: https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    value = inMax - value if reverse else value  # Reversing the range
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


def get_palette(library='mpl', name='viridis', n_colors=None):
    if library == 'mpl':
        cmap = mpl.colormaps[name]
        if n_colors:
            colors = cmap(np.linspace(0, 1, n_colors))
        else:
            # Qualitative color map with predefined number of colors
            colors = cmap.colors
            # prop_cycle = plt.rcParams['axes.prop_cycle']
            # colors = prop_cycle.by_key()['color']
        colors = [mpl.colors.rgb2hex(c) for c in colors]

    elif library == 'sns':
        colors = sns.color_palette(name, n_colors)
        colors = [mpl.colors.rgb2hex(c) for c in colors]

    elif library == 'glasbey':
        colors = glasbey.create_palette(palette_size=n_colors)
    else:
        raise Exception(
            'Only the following libraries are supported: mpl, sns or glasbey.'
        )

    return colors


def labels2colors(X, y, mapping, imbalanced=False, cmap_sequential=False):
    # For scatter plots with categorical variables: map each label to a color
    n_colors = len(mapping) - 1 if imbalanced else len(mapping)
    colors = (
        get_palette(n_colors=n_colors)
        if cmap_sequential
        else get_palette(library='glasbey', n_colors=n_colors)
    )

    # Assign gray to the majority class which will be the first on the mapping
    if imbalanced:
        colors.insert(0, '#7f7f7f')

    # Sort labels by descending order
    keys, counts = np.unique(y, return_counts=True)
    keys = keys[np.argsort(counts)[::-1]]

    color_mapping = OrderedDict()
    c = np.empty_like(y, dtype=object)
    for k, color in zip(keys, colors):
        c[y == k] = color
        # Assign each color to its label name instead of the label (digitized) value
        color_mapping[mapping[k]] = color

    return X, c, color_mapping


# def sample_imbalanced(X, y):
#     # Undersample imbalanced variables, nans should not be present

#     # Only undersample the majority class to the sum of the minority classes
#     keys, values = np.unique(y, return_counts=True)
#     max_idx, max_val = values.argmax(), values.max()
#     values[max_idx] = 0
#     values[max_idx] = np.min([values.sum(), max_val])
#     sampling = dict(zip(keys, values))

#     rus = RandomUnderSampler(sampling_strategy=sampling, random_state=42)
#     X_res, y_res = rus.fit_resample(X, y)
#     return X_res, y_res


def drop_nans(X, y, mapping=None):
    if np.isnan(y).any():
        X = X[~np.isnan(y), :]
        y = y[~np.isnan(y)]
    if mapping is not None:
        mapping.pop(np.nan, None)
    return X, y, mapping


def plot_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    mappings: dict,
    plot_file=None,
    n_subplots=(4, 3),
    fig_width='full',
    fig_height_ratio=0.8,
    titles=[],
    imbalanced=None,
    categorical=None,
    cbar_shrink=1,
    cbar_location='right',
    s_marker=None,
    return_ax=False,
):
    # X must be of shape (n, 2), y must be of shape (n, n_features)
    categorical = [None] * y.shape[1] if not categorical else categorical
    imbalanced = [None] * y.shape[1] if not imbalanced else imbalanced

    assert len(categorical) == y.shape[1], (
        'categorical must be None or a list with boolean values with shape y.shape[1].'
    )
    assert len(imbalanced) == y.shape[1], (
        'imbalanced must be None or a list with boolean values with shape y.shape[1].'
    )

    fig, ax = plt.subplots(*n_subplots)
    set_figsize(fig, fig_width, height_ratio=fig_height_ratio)
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    # Plot each feature
    for i in range(y.shape[1]):
        X_i, y_i, mapping = drop_nans(np.copy(X), y[:, i], mappings[i].copy())

        # Use undersampling
        # if imbalanced[i]:
        #     X_i, y_i = sample_imbalanced(X_i, y_i)

        if categorical[i]:
            X_c, c, color_mapping = labels2colors(
                X_i, y_i, mapping, imbalanced=imbalanced[i]
            )
            alpha = (
                [0.5] + [1] * (len(color_mapping) - 1)
                if imbalanced[i]
                else [0.5] * len(color_mapping)
            )  # If imbalanced grey markers have higher transparency

            for j, (k, v) in enumerate(color_mapping.items()):
                ax[i].scatter(
                    *X_c[c == v, :].T,
                    c=v,
                    s=map_range(X_c.shape[0]) if s_marker is None else s_marker,
                    alpha=alpha[j],
                    marker='.',
                    linewidths=0,
                    edgecolors='None',
                    label=k,
                )

            box = ax[i].get_position()
            ax[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax[i].legend(
                loc='center left',
                markerscale=map_range(X_c.shape[0], outMin=2, outMax=4, reverse=False),
                bbox_to_anchor=(1, 0.5),
            )
            _ = [lh.set_alpha(1.0) for lh in legend.legend_handles]
            ax[i].set_title(titles[i], loc='center', wrap=True)
            ax[i].axis('equal')
            ax[i].set_axis_off()

        else:
            p = ax[i].scatter(
                *X_i.T,
                c=y_i,
                s=map_range(X_i.shape[0]) if s_marker is None else s_marker,
                alpha=0.5,
                marker='.',
                linewidths=0,
                edgecolors='None',
                cmap='viridis_r',
            )
            cbar = fig.colorbar(
                p, shrink=cbar_shrink, location=cbar_location, pad=0, ax=ax[i]
            )
            cbar.solids.set(alpha=1)
            ax[i].set_title(titles[i], loc='center', wrap=True)
            ax[i].axis('equal')
            ax[i].set_axis_off()

    plt.tight_layout()
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight', dpi=600, format='svg')

    if return_ax:
        return fig, ax


def annotate_embeddings_plot(
    fig, ax, imgs, img_labels, img_titles, embedding_coords, img_coords, plot_file=None
):
    arrowprops = dict(
        arrowstyle='-',
        linewidth=plt.rcParams['axes.linewidth'],
        color='xkcd:slate gray',
    )

    for img, img_label, img_title, emb_xy, im_xy in zip(
        imgs, img_labels, img_titles, embedding_coords, img_coords
    ):
        border_color = ax.collections[0].to_rgba(img_label)[:-1]
        border_color = tuple(int(c * 255) for c in border_color)
        im_with_border = np.array(ImageOps.expand(img, border=4, fill=border_color))

        txt = mpl.offsetbox.TextArea(img_title, textprops=dict(size=10))
        imbox = mpl.offsetbox.OffsetImage(im_with_border, zoom=0.25)
        pack = mpl.offsetbox.VPacker(children=[txt, imbox], sep=1, align='center')
        abox = mpl.offsetbox.AnnotationBbox(
            pack, emb_xy, im_xy, arrowprops=arrowprops, frameon=False, pad=-0.2
        )

        ax.add_artist(abox)

    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')

    return fig, ax


def get_rotation_matrix(theta):
    # Compute 2D rotation matrix
    theta_rad = theta * np.pi / 180
    transform = np.float32(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )
    return transform


if __name__ == '__main__':
    pass

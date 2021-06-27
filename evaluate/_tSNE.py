import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
# import matplotlib.font_manager
# from matplotlib import rcParams
import matplotlib.pyplot as plt


# # enable latex support
# matplotlib.rcParams['text.usetex'] = True

# # use Times New Roman
# rcParams['font.family'] = 'Times New Roman'

# # use Times New Roman NOT bold
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()


def t_sne(F, L, title="tsne", path="log"):
    """T-SNE visualization
    F: [n, d], features
    L: [n], label id
    """
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    F = tsne.fit_transform(F)
    fig = plt.figure()
    plt.title(title)
    l1 = plt.scatter(F[:, 0], F[:, 1], s=25, c=L, marker='.', cmap="rainbow")
    plt.legend(handles=[l1], labels=[title], loc="best")
    # plt.show()
    fig.savefig(os.path.join(path, "{}.png".format(title)))
    plt.close(fig)

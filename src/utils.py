import os

import tikzplotlib as tpl


def rotate_list(xs):
    while True:
        for x in xs:
            yield x


def save_fig(figure, dir_path, name, tex_subdir=None):
    if tex_subdir is not None:
        tex_dir = os.path.join(dir_path, tex_subdir)
        deep_dir = tex_dir
    else:
        deep_dir = dir_path
    if not os.path.exists(deep_dir):
        os.makedirs(deep_dir)
    figure.savefig(f'{dir_path}/{name}.pdf', dpi=500)
    if tex_subdir is not None:
        tpl.save(f'{tex_dir}/{name}.tex',
                 figure=figure,
                 axis_width=r'\figwidth',
                 axis_height=r'\figheight')

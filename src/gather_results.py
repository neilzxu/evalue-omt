import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ProgramName',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument('dir_path')  # positional argument
    parser.add_argument('--out_dir', required=True)  # positional argument
    args = parser.parse_args()

    names = os.listdir(args.dir_path)
    dfs = [pd.read_csv(os.path.join(args.dir_path, name)) for name in names]
    df = pd.concat(dfs, axis=0).reset_index()
    #df = df.iloc[:2400, :]

    df = df.fillna(0)
    figure_dir = args.out_dir
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    print(df)
    df = df.rename(mapper={
        'power': 'Power',
        'FDP': 'FDR',
        'method': 'Method'
    },
                   axis='columns')

    sns.set_theme()
    plt.rcParams.update({'lines.markeredgewidth': 1})
    for metric in ['FDR', 'Power']:
        g = sns.catplot(data=df,
                        kind="bar",
                        x="score",
                        y=metric,
                        hue="Method",
                        errorbar=("se", 1),
                        palette="dark",
                        alpha=.6,
                        height=4,
                        capsize=.15,
                        err_kws={'linewidth': 1.})
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_xlabel('')
        g.despine(left=True)
        g.savefig(f'{figure_dir}/{metric}.pdf')

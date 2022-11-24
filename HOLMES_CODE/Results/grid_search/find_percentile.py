import os, glob, argparse, json, tqdm
import numpy as np


def find_best_percentile(input_folder):

    avg_insertions = []
    avg_deletions = []
    avg_preservations = []
    percentiles = []

    for json_f in glob.glob(input_folder + '/*.json'):

        perc = json_f.split('_')[-2]
        percentiles.append(perc)

        insertions = []
        deletions = []
        preservations = []

        with open(json_f) as json_file:
            stats_per_holonym = json.load(json_file)

        for holonym, hol_stats in tqdm.tqdm(stats_per_holonym.items()):
            hol_curves = hol_stats["curves"]

            for img_stats in hol_curves.values():                

                img_del = img_stats["holmes_deletion"]
                img_ins = img_stats["holmes_insertion"]
                img_pres = img_stats["holmes_preservation"]

                insertions.append(img_ins)
                deletions.append(img_del)
                preservations.append(img_pres)

        avg_insertions.append(np.mean(insertions))
        avg_deletions.append(np.mean(deletions))
        avg_preservations.append(np.mean(preservations))

    # now I have all curves, take best

    avg_insertions = np.around(avg_insertions, decimals=3, out=None)
    ins_best = np.argmax(avg_insertions)
    ins_mark = ['  X  ' if i == ins_best else '' for i, x in enumerate(avg_insertions)]
    avg_deletions = np.around(avg_deletions, decimals=3, out=None)
    del_best = np.argmin(avg_deletions)
    del_mark = ['  X  ' if i == del_best else '' for i, x in enumerate(avg_deletions)]
    avg_preservations = np.around(avg_preservations, decimals=3, out=None)
    pres_best = np.argmax(avg_preservations)
    pres_mark = ['  X  ' if i == pres_best else '' for i, x in enumerate(avg_preservations)]

    idx_sum = np.zeros(len(avg_insertions))
    for i in range(len(idx_sum)):
        if i == ins_best:
            idx_sum[i] += 1
        if i == del_best:
            idx_sum[i] += 1
        if i == pres_best:
            idx_sum[i] += 1

    #best_percs = np.argwhere(idx_sum == np.amax(idx_sum)).flatten()
    best_percs = np.argwhere(idx_sum > 0).flatten()
    best_percs_X = ['  X  ' if i in best_percs else '' for i, x in enumerate(avg_insertions)]

    print()
    print(f"                {str(['  {0: <3}'.format(str(x)) for x in percentiles])}")
    print(f"INSERTIONS    : {str(['{0: <5}'.format(str(x)) for x in avg_insertions])}")
    print(f"              : {str(['{0: <5}'.format(str(x)) for x in ins_mark])}")
    print(f"DELETIONS     : {str(['{0: <5}'.format(str(x)) for x in avg_deletions])}")
    print(f"              : {str(['{0: <5}'.format(str(x)) for x in del_mark])}")
    print(f"PRESERVATIONS : {str(['{0: <5}'.format(str(x)) for x in avg_preservations])}")
    print(f"              : {str(['{0: <5}'.format(str(x)) for x in pres_mark])}")
    print()
    print(f"BEST PERC(S)  : {str(['{0: <5}'.format(str(x)) for x in best_percs_X])}")
    print()
    best_percs_found = [int(percentiles[i]) for i, x in enumerate(avg_insertions) if i in best_percs]
    print(f"BEST PERC(S) FOUND (CHOOSE ONE IN BETWEEN): {best_percs_found}")
    print(f"SUGGESTED VALUE: {int(np.around(np.mean(best_percs_found)+1e-7))}")




def main():
    parser = argparse.ArgumentParser(description='Find best percentile based on curves metrics.')
    parser.add_argument('--input_folder', type=str, default='.', help='folder where curves stats per percentile are stored.', required=False)
    args = parser.parse_args()

    input_folder = args.input_folder

    find_best_percentile(input_folder)


if __name__ == "__main__":
    main()
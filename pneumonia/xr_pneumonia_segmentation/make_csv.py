import os

import pandas as pd


def create_hyper_csv(fname='./jmodels/hyper.csv', overwrite=False):
    if os.path.exists(fname) and not overwrite:
        return
    id = 0
    df = {'output_dir': [], 'fold': [], 'batch_size': [], 'epochs': [], 'filters': [], 'block_scale': [], 'alpha': [],
          'beta': [], 'use_mask': []}
    for alpha, beta in [(0.3, 1.0)]:
        for epochs in [200]:
            for filters in [16, 32]:
                for block_scale in [1, 2]:
                    for use_mask in [0, 1]:
                        # --- Create exp
                        for fold in range(1):
                            df['output_dir'].append('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id))
                            df['fold'].append(fold)
                            df['batch_size'].append(8)
                            df['epochs'].append(epochs)
                            df['filters'].append(filters)
                            df['block_scale'].append(block_scale)
                            df['alpha'].append(alpha)
                            df['beta'].append(beta)
                            df['use_mask'].append(use_mask)
                            os.makedirs('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id), exist_ok=True)
                            id += 1

    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)
    print('Created {} successfully'.format(fname))


create_hyper_csv(overwrite=True)
os.system('rm -rf "./scripts"')
os.system('jarvis script -jmodels ./jmodels -name xr_pna_seg -output_dir "./scripts"')

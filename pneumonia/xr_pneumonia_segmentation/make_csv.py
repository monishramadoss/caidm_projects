import os

import pandas as pd

def create_hyper_csv(fname='./jmodels/hyper.csv', overwrite=False):
    if os.path.exists(fname) and not overwrite:
        return
    id = 0
    df = {'output_dir': [], 'fold': [], 'batch_size': [], 'epochs': [], 'filters': [], 'block_scale': [], 'alpha': [],
          'beta': [], 'use_mask': [],'scalei':[], 'scaleo':[],
          'block1': [],'block2': [],'block3': [],'block4': [],'block5': [],
          'scale1':[], 'scale2':[], 'scale3':[], 'scale4':[], 'scale5':[] }
    
    for alpha, beta in [(0.3, 1.0)]:
        for epochs in [200]:
            for filters in [32]:
                for block_scale in [1]:
                    for use_mask in [0, 1]:
                        for scalei in [8]:
                            for scaleo in [6]:
                                for scale1, scale2, scale3, scale4, scale5 in [(1, 1, 2, 3, 4)]:
                                    for block1, block2, block3, block4, block5 in [(4, 4, 4, 4, 4), (4, 1, 2, 3, 4)]:
                                        # --- Create exp
                                        for fold in range(4):
                                            df['output_dir'].append('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id))
                                            df['fold'].append(fold)
                                            df['batch_size'].append(8)
                                            df['epochs'].append(epochs)
                                            df['filters'].append(filters)
                                            df['block_scale'].append(block_scale)
                                            df['alpha'].append(alpha)
                                            df['beta'].append(beta)
                                            df['use_mask'].append(use_mask)
                                            df['scalei'].append(scalei)
                                            df['scaleo'].append(scaleo)
                                            
                                            df['scale1'].append(scale1)
                                            df['scale2'].append(scale2)
                                            df['scale3'].append(scale3)
                                            df['scale4'].append(scale4)
                                            df['scale5'].append(scale5)
                                            
                                            df['block1'].append(block1)
                                            df['block2'].append(block2)
                                            df['block3'].append(block3)
                                            df['block4'].append(block4)
                                            df['block5'].append(block5)

                                            os.makedirs('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id), exist_ok=True)
                                            id += 1

    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)
    print('Created {} successfully'.format(fname))


create_hyper_csv(overwrite=True)
os.system('rm -rf "./scripts"')
os.system('jarvis script -jmodels ./jmodels -name xr_pna_seg -output_dir "./scripts"')

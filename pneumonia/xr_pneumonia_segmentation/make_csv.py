import os

import pandas as pd

def create_hyper_csv(fname='./jmodels/hyper.csv', overwrite=False):
    if os.path.exists(fname) and not overwrite:
        return
    id = 0
    df = {'output_dir': [], 'fold': [], 'filters': [], 'alpha': [],
          'beta': [], 'use_mask': [],'scalei':[], 'scaleo':[],          
          'scale1':[], 'scale2':[], 'scale3':[], 'scale4':[], 'scale5':[] }
    
    for alpha, beta in [(0.3, 1.0)]:
        for filters in [16]:
            for use_mask in [0, 1]:
                for scalei in [8]:
                    for scaleo in [6]:
                        for scale1, scale2, scale3, scale4, scale5 in [(1, 1, 2, 3, 4)]:
                            # --- Create exp
                            for fold in range(4):
                                df['output_dir'].append('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id))
                                df['fold'].append(fold)
                                df['filters'].append(filters)
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
                                
                                os.makedirs('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id), exist_ok=True)
                                id += 1

    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)
    print('Created {} successfully'.format(fname))


create_hyper_csv(overwrite=True)
os.system('rm -rf "./scripts"')
os.system('jarvis script -jmodels ./jmodels -name xr_pna_seg -output_dir "./scripts"')

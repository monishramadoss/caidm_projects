import os
import pandas as pd
import shutil
def create_hyper_csv(fname='./jmodels/hyper.csv', overwrite=False):
    if os.path.exists(fname) and not overwrite:
        return
    if(not os.path.isdir('./jmodels/exp')):
        os.makedirs('./jmodels/exp')
    else:
        shutil.rmtree('./jmodels/exp')
        os.makedirs('./jmodels/exp')
        
    id = 0
    df = {
        'output_dir': [], 
        'fold': [],
        'batch_size': [],
        'epochs': [],
        'filters1':[],
        'filters2':[],
        'block_scale1':[],
        'block_scale2':[],
        'alpha':[],
        'beta':[],
        'gamma':[],
        'delta':[]
    }
    
    for alpha, beta in [(0.3, 1.0), (1.0, 1.0)]:
        for gamma, delta in [(0.3, 1.0), (1.0, 1.0)]:
            for epochs in [100]:
                for filters1, filters2 in [(32,32)]:
                    for block_scale1, block_scale2 in [(1,1), (1,2)]:
                        for fold in range(1):
                            df['output_dir'].append('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id) )
                            df['fold'].append(fold)
                            df['batch_size'].append(8)
                            df['epochs'].append(epochs)
                            df['filters1'].append(filters1)
                            df['filters2'].append(filters2)
                            df['block_scale1'].append(block_scale1)
                            df['block_scale2'].append(block_scale2)

                            df['alpha'].append(alpha)
                            df['beta'].append(beta)
                            df['gamma'].append(gamma)
                            df['delta'].append(delta)
                            os.makedirs('{0}/jmodels/exp/exp_{1}'.format(os.getcwd(), id), exist_ok=True)
                            id+=1
    print(id)
    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)    
    print('Created {} successfully'.format(fname))

create_hyper_csv(overwrite=True)
os.system('rm -rf "./scripts"')
os.system('jarvis script -jmodels ./jmodels -name xr_pna_seg -output_dir "./scripts"')
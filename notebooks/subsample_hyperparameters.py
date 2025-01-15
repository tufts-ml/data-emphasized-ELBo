import itertools
import numpy as np
import pandas as pd

def subsample_hyperparameters(ns, retrained_df, tuned_df, random_states=[1001, 2001, 3001]):
    
    dictionary = {
        n: {
            random_state: np.zeros((100, 144, 500, 2)) for random_state in random_states
        } for n in ns
    }
    
    for n, random_state in itertools.product(ns, random_states):
        for subsample_round_index in range(100):
            for num_hypers in range(1,145):
                for subsample_index in range(500):
                    condition = (tuned_df.criterion=='l2-sp')&(tuned_df.n==n)&(tuned_df.random_state==random_state)
                    sampled_tuned_df = tuned_df.loc[condition].sample(n=num_hypers)
                    runtime = sampled_tuned_df.runtime.sum()
                    best_model_name = sampled_tuned_df.loc[sampled_tuned_df.val_nll.idxmin(), 'model_name']
                    test_acc = retrained_df.loc[retrained_df.model_name==best_model_name, 'test_acc'].item()
                    runtime += retrained_df.loc[retrained_df.model_name==best_model_name, 'runtime'].item()
                    dictionary[n][random_state][subsample_round_index,num_hypers-1,subsample_index,0] = runtime
                    dictionary[n][random_state][subsample_round_index,num_hypers-1,subsample_index,1] = test_acc

    return dictionary

if __name__=='__main__':
    #tuned_cifar10_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/tuned_CIFAR-10_ConvNeXt_Tiny.csv')
    #retrained_cifar10_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/retrained_CIFAR-10_ConvNeXt_Tiny.csv')
    #cifar10_dictionary = subsample_hyperparameters([100, 1000, 10000, 50000], retrained_cifar10_df, tuned_cifar10_df)
    #np.save('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/CIFAR-10_ConvNeXt_Tiny_dictionary.npy', cifar10_dictionary)
    
    #tuned_oxfordiiit_pet_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/tuned_Oxford-IIIT_Pet_ConvNeXt_Tiny.csv')
    #retrained_oxfordiiit_pet_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/retrained_Oxford-IIIT_Pet_ConvNeXt_Tiny.csv')
    #oxfordiiit_pet_dictionary = subsample_hyperparameters([370, 3441], retrained_oxfordiiit_pet_df, tuned_oxfordiiit_pet_df)
    #np.save('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/Oxford-IIIT_Pet_ConvNeXt_Tiny_dictionary.npy', oxfordiiit_pet_dictionary)
    
    tuned_flowers_102_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/tuned_Flowers_102_ConvNeXt_Tiny.csv')
    retrained_flowers_102_df = pd.read_csv('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/retrained_Flowers_102_ConvNeXt_Tiny.csv')
    flowers_102_dictionary = subsample_hyperparameters([510, 1020], retrained_flowers_102_df, tuned_flowers_102_df)
    np.save('/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/notebooks/Flowers_102_ConvNeXt_Tiny_dictionary.npy', flowers_102_dictionary)
    
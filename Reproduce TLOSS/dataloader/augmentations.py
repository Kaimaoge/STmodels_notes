import numpy as np
import torch


def Data_Transform(data, configs):
    batch_size = data.size(0)
    length = data.size(2)
      
    samples = np.concatenate(
            [np.random.choice(
                np.delete(np.arange(batch_size), j),
                size=(configs.nb_random_samples, 1)
            ) for j in range(batch_size)], axis = 1
        )                 # It can perfectly avoid self selection
    
    samples = torch.LongTensor(samples)
    length_pos_neg = np.random.randint(1, high=length + 1)
    random_length = np.random.randint(
        length_pos_neg, high=length + 1
    )  
    beginning_batches = np.random.randint(
        0, high=length - random_length + 1, size=batch_size
    ) 
    beginning_samples_pos = np.random.randint(
        0, high=random_length - length_pos_neg + 1, size=batch_size
    )  
    
    beginning_positive = beginning_batches + beginning_samples_pos
    end_positive = beginning_positive + length_pos_neg
    
    beginning_samples_neg = np.random.randint(
        0, high=length - length_pos_neg + 1,
        size=(configs.nb_random_samples, batch_size)
    )
    
    x_ref = torch.cat(
            [data[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        )
    
    x_pos = torch.cat(
            [data[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        )
    
    x_neg = []
    for i in range(configs.nb_random_samples):
        x_neg.append(torch.cat([data[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)]))
            
    return x_ref, x_pos, x_neg

    

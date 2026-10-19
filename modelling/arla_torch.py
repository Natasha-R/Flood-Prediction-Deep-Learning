import torch

def arla(label, # label (or model output)
         sensitivity=1, # the percentage threshold to aggregate the (sub)patch as the positive class
         subpatches=1, # the number of subpatches to aggregate the overall patch into
         positive_class=1, # the positive class integer
         negative_class=0, # the negative class integer
         invalid_class=-1, # the class integer representing non-data pixels
         multiclass=False): # binary or multi-class segmentation

    label = label.clone()
    splits = torch.tensor_split(torch.arange(label.shape[1], device=label.device), subpatches)
    label_shape = len(label.shape)
    if label_shape == 2:
        label = label.unsqueeze(0)
    
    for index in range(label.shape[0]):
        for row_index in splits:
            for col_index in splits:

                start_row, end_row = row_index[0].item(), row_index[-1].item() + 1
                start_col, end_col = col_index[0].item(), col_index[-1].item() + 1
                subpatch = label[index, start_row:end_row, start_col:end_col]
                original_subpatch = subpatch.clone()

                if not multiclass:
                    percentage = (float((subpatch == positive_class).sum()) / (subpatch != invalid_class).sum().float()) * 100
                    subpatch[:] = positive_class if percentage >= sensitivity else negative_class 
                else:
                    subpatch[:] = torch.argmax(torch.bincount(subpatch.flatten().long()))
                subpatch[original_subpatch == invalid_class] = invalid_class

    if label_shape == 2:
        label = label.squeeze()

    return label
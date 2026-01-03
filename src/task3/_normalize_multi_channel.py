class NormalizeMultiChannel:
    """Normalizes a tensor channel-wise to the range [0, 1]."""
    def __call__(self, tensor):
        # tensor shape: (C, H, W)
        for i in range(tensor.shape[0]):
            t_min = tensor[i].min()
            t_max = tensor[i].max()
            if t_max - t_min > 0:
                tensor[i] = (tensor[i] - t_min) / (t_max - t_min)
        return tensor
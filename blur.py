
import torch.nn.functional as F

def gaussian_blur(input_tensor, kernel_size, sigma=None):
    if sigma is None:
        sigma = (kernel_size - 1) / 6  # Impute ideal sigma from kernel size

    channels = input_tensor.shape[1]  # Get number of channels from input tensor

    # Create Gaussian kernel
    kernel = torch.tensor([
        [((x - kernel_size // 2) / sigma)**2 for x in range(kernel_size)]
        for y in range(kernel_size)
    ])
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    # Apply Gaussian blur
    blurred = F.conv2d(input_tensor, kernel, stride=1, padding=kernel_size // 2)

    return blurred

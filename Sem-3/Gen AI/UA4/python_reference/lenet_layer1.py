import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetLayer1(nn.Module):
    """
    First convolutional layer of LeNet-5
    Input: 28x28x1 (MNIST)
    Conv1: 6 filters, 5x5 kernel, stride=1, padding=2
    Output: 28x28x6
    Activation: ReLU (or Tanh in original)
    """
    def __init__(self):
        super(LeNetLayer1, self).__init__()
        # First conv layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
                               kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        return F.relu(self.conv1(x))

def save_weights_and_test():
    """Generate test data and save weights for Verilog verification"""
    
    # Create model and set to eval mode
    model = LeNetLayer1()
    model.eval()
    
    # Create simple test input (28x28)
    test_input = np.random.randint(0, 256, (1, 1, 28, 28)).astype(np.float32) / 255.0
    
    # Save test input
    np.save('test_data/layer1_input.npy', test_input)
    
    # Get output
    with torch.no_grad():
        input_tensor = torch.from_numpy(test_input)
        output = model(input_tensor)
        output_np = output.numpy()
    
    # Save output
    np.save('test_data/layer1_output.npy', output_np)
    
    # Save weights and biases
    weights = model.conv1.weight.detach().numpy()  # Shape: (6, 1, 5, 5)
    biases = model.conv1.bias.detach().numpy()      # Shape: (6,)
    
    np.save('test_data/layer1_weights.npy', weights)
    np.save('test_data/layer1_biases.npy', biases)
    
    print("Layer 1 Configuration:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Biases shape: {biases.shape}")
    print(f"\nInput range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
    
    # Save as text files for easy Verilog reading
    save_as_hex(test_input[0, 0], 'test_data/layer1_input.hex')
    save_as_hex(weights, 'test_data/layer1_weights.hex')
    save_as_hex(biases, 'test_data/layer1_biases.hex')
    save_as_hex(output_np[0], 'test_data/layer1_output.hex')
    
    return model, test_input, output_np

def save_as_hex(data, filename):
    """Convert float data to fixed-point hex for Verilog"""
    # Use Q8.8 fixed point format (8 integer bits, 8 fractional bits)
    fixed_point = (data * 256).astype(np.int32)
    
    with open(filename, 'w') as f:
        flat_data = fixed_point.flatten()
        for val in flat_data:
            # Convert to 16-bit signed hex
            if val < 0:
                val = val & 0xFFFF
            f.write(f"{val:04x}\n")

if __name__ == "__main__":
    import os
    os.makedirs('test_data', exist_ok=True)
    
    model, input_data, output_data = save_weights_and_test()
    print("\nTest data generated successfully!")
    print("Files saved in test_data/ directory")

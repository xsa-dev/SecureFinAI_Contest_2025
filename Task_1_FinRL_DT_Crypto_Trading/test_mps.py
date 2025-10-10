#!/usr/bin/env python3
"""
Test script for MPS (Metal Performance Shaders) support on Apple Silicon.
This script tests the device utilities and runs a simple training example.
"""

import torch
import numpy as np
import time
from device_utils import get_device, print_device_info, get_device_info


def test_device_selection():
    """Test device selection and display information."""
    print("🧪 Testing Device Selection")
    print("=" * 50)
    
    # Print device information
    print_device_info()
    
    # Test device selection
    device = get_device(gpu_id=-1, verbose=True)
    print(f"\n✅ Selected device: {device}")
    
    return device


def test_tensor_operations(device):
    """Test basic tensor operations on the selected device."""
    print(f"\n🧮 Testing Tensor Operations on {device}")
    print("=" * 50)
    
    # Create test tensors
    size = 1000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Test matrix multiplication
    start_time = time.time()
    c = torch.matmul(a, b)
    end_time = time.time()
    
    print(f"Matrix multiplication ({size}x{size}): {end_time - start_time:.4f} seconds")
    
    # Test memory usage
    if device.type == "mps":
        print("💾 MPS Memory Info:")
        print(f"   - MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
        print(f"   - MPS memory cached: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
    elif device.type == "cuda":
        print("💾 CUDA Memory Info:")
        print(f"   - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"   - CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return c


def test_decision_transformer_components(device):
    """Test Decision Transformer related components."""
    print(f"\n🤖 Testing Decision Transformer Components on {device}")
    print("=" * 50)
    
    try:
        from transformers import DecisionTransformerConfig, DecisionTransformerModel
        
        # Create a small Decision Transformer model
        config = DecisionTransformerConfig(
            state_dim=10,
            act_dim=3,
            hidden_size=64,
            max_ep_len=20,
            n_positions=20,
            n_layer=2,
            n_head=1,
        )
        
        model = DecisionTransformerModel(config).to(device)
        
        # Test forward pass
        batch_size = 4
        seq_len = 20
        
        states = torch.randn(batch_size, seq_len, config.state_dim, device=device)
        actions = torch.randint(0, config.act_dim, (batch_size, seq_len), device=device)
        returns_to_go = torch.randn(batch_size, seq_len, 1, device=device)
        timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        print(f"Input shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Returns to go: {returns_to_go.shape}")
        print(f"  Timesteps: {timesteps.shape}")
        print(f"  Attention mask: {attention_mask.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            outputs = model(
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
        end_time = time.time()
        
        print(f"Forward pass completed in: {end_time - start_time:.4f} seconds")
        print(f"Output logits shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Decision Transformer: {e}")
        return False


def test_training_simulation(device):
    """Test a simple training simulation."""
    print(f"\n🎯 Testing Training Simulation on {device}")
    print("=" * 50)
    
    try:
        # Simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        num_epochs = 10
        batch_size = 32
        
        print(f"Training for {num_epochs} epochs with batch size {batch_size}")
        
        for epoch in range(num_epochs):
            # Generate random data
            x = torch.randn(batch_size, 10, device=device)
            y = torch.randn(batch_size, 3, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        print("✅ Training simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in training simulation: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 MPS Support Test for SecureFinAI Contest 2025 - Task 1")
    print("=" * 70)
    
    # Test device selection
    device = test_device_selection()
    
    # Test tensor operations
    test_tensor_operations(device)
    
    # Test Decision Transformer components
    dt_success = test_decision_transformer_components(device)
    
    # Test training simulation
    training_success = test_training_simulation(device)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Decision Transformer: {'✅ Pass' if dt_success else '❌ Fail'}")
    print(f"Training Simulation: {'✅ Pass' if training_success else '❌ Fail'}")
    
    if dt_success and training_success:
        print("\n🎉 All tests passed! MPS support is working correctly.")
        print("💡 You can now run the full Task 1 pipeline with MPS acceleration.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("💡 You can still run the pipeline on CPU if needed.")
    
    return dt_success and training_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Complete Task 1 pipeline runner with MPS support for Apple Silicon.
This script runs the entire crypto trading pipeline with optimal device selection.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from device_utils import get_device, print_device_info


def run_command(cmd, cwd=None, description=""):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            check=True, 
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking Requirements")
    print("=" * 50)
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Check MPS
        if torch.backends.mps.is_available():
            print("✅ MPS: Available")
        else:
            print("⚠️  MPS: Not available (will use CPU)")
            
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available ({torch.cuda.device_count()} devices)")
        else:
            print("ℹ️  CUDA: Not available")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check other dependencies
    try:
        import pandas
        import numpy
        import sklearn
        from transformers import DecisionTransformerConfig
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    return True


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories")
    print("=" * 50)
    
    dirs = [
        "data",
        "trained_models", 
        "plots",
        "offline_data_preparation/data",
        "offline_data_preparation/output"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")


def run_data_preparation():
    """Run data preparation steps."""
    print("\n📊 Data Preparation")
    print("=" * 50)
    
    # Check if data exists
    data_file = "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv"
    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        print("Please download the required datasets:")
        print("1. BTC_1sec_with_sentiment_risk_train.csv from Google Drive")
        print("2. FinRL_BTC_news_signals from Hugging Face")
        return False
    
    # Run data processing
    steps = [
        ("python seq_data.py", "Processing BTC data and generating Alpha101 factors"),
        ("python seq_run.py 0", "Training RNN for factor aggregation"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, cwd="offline_data_preparation", description=desc):
            return False
    
    return True


def run_rl_training():
    """Run reinforcement learning training."""
    print("\n🤖 RL Training")
    print("=" * 50)
    
    # Run RL training
    steps = [
        ("python erl_run.py 0", "Training single DQN agent"),
        ("python task1_ensemble.py 0", "Training ensemble RL agents"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, cwd="offline_data_preparation", description=desc):
            print(f"⚠️  {desc} failed, continuing...")
    
    return True


def run_trajectory_conversion():
    """Convert RL trajectories to Decision Transformer format."""
    print("\n🔄 Trajectory Conversion")
    print("=" * 50)
    
    # Find replay buffer directory
    replay_dirs = [d for d in os.listdir("offline_data_preparation") if d.startswith("TradeSimulator")]
    
    if not replay_dirs:
        print("❌ No replay buffer directories found")
        return False
    
    replay_dir = replay_dirs[0]  # Use the first one found
    print(f"Using replay buffer: {replay_dir}")
    
    cmd = f"python convert_replay_buffer_to_trajectories.py --replay_buffer_dir ./{replay_dir} --output_file ../crypto_trajectories.csv"
    
    if not run_command(cmd, cwd="offline_data_preparation", description="Converting trajectories"):
        return False
    
    return True


def run_decision_transformer():
    """Run Decision Transformer training."""
    print("\n🤖 Decision Transformer Training")
    print("=" * 50)
    
    # Check if trajectory data exists
    if not os.path.exists("crypto_trajectories.csv"):
        print("❌ Trajectory data not found. Run trajectory conversion first.")
        return False
    
    # Run Decision Transformer training with MPS-optimized parameters
    cmd = "python dt_crypto.py --epochs 50 --lr 1e-3 --context_length 10 --model_path ./trained_models/decision_transformer.pth --plots_dir plots"
    
    if not run_command(cmd, description="Training Decision Transformer"):
        return False
    
    return True


def run_evaluation():
    """Run model evaluation."""
    print("\n📊 Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = "trained_models/decision_transformer.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Run evaluation
    cmd = f"python evaluation.py --model_path {model_path} --test_data_path ./offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv --max_samples 1000 --target_return 250.0 --context_length 10 --plots_dir plots"
    
    if not run_command(cmd, description="Evaluating model"):
        return False
    
    return True


def main():
    """Main pipeline runner."""
    print("🚀 SecureFinAI Contest 2025 - Task 1 Pipeline with MPS Support")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return False
    
    # Print device information
    print_device_info()
    
    # Setup directories
    setup_directories()
    
    # Run pipeline steps
    steps = [
        ("Data Preparation", run_data_preparation),
        ("RL Training", run_rl_training),
        ("Trajectory Conversion", run_trajectory_conversion),
        ("Decision Transformer", run_decision_transformer),
        ("Evaluation", run_evaluation),
    ]
    
    start_time = time.time()
    
    for step_name, step_func in steps:
        print(f"\n{'='*70}")
        print(f"🎯 {step_name}")
        print(f"{'='*70}")
        
        step_start = time.time()
        
        if not step_func():
            print(f"❌ {step_name} failed")
            print("💡 You can continue with the next step or fix the issue and rerun")
            
            # Ask if user wants to continue
            response = input("\nContinue with next step? (y/n): ").lower()
            if response != 'y':
                break
        else:
            step_time = time.time() - step_start
            print(f"✅ {step_name} completed in {step_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Pipeline completed in {total_time:.2f} seconds")
    
    # Print results summary
    print("\n📊 Results Summary")
    print("=" * 50)
    
    if os.path.exists("trained_models/decision_transformer.pth"):
        print("✅ Decision Transformer model trained")
    
    if os.path.exists("plots"):
        plot_files = list(Path("plots").glob("*.png"))
        if plot_files:
            print(f"✅ Generated {len(plot_files)} plots")
    
    print("\n💡 Next steps:")
    print("1. Check the plots/ directory for training visualizations")
    print("2. Review the trained model in trained_models/")
    print("3. Analyze the evaluation results")
    print("4. Experiment with different hyperparameters")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from stable_baselines3 import PPO
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.plot import backtest_stats
from stable_baselines3.common.logger import configure

def create_enhanced_data(data_file='train_data.csv'):
    """
    Create enhanced dataset with additional features
    """
    processed_full = pd.read_csv(data_file)
    
    # Add additional technical indicators
    for window in [5, 10, 20, 50]:
        processed_full[f'close_{window}_sma'] = processed_full.groupby('tic')['close'].rolling(window=window).mean().reset_index(0, drop=True)
        processed_full[f'volume_{window}_sma'] = processed_full.groupby('tic')['volume'].rolling(window=window).mean().reset_index(0, drop=True)
    
    # Price momentum indicators
    processed_full['price_momentum_5'] = processed_full.groupby('tic')['close'].pct_change(5)
    processed_full['price_momentum_10'] = processed_full.groupby('tic')['close'].pct_change(10)
    
    # Volume indicators
    processed_full['volume_ratio'] = processed_full['volume'] / processed_full.groupby('tic')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
    
    # Volatility indicators
    processed_full['volatility_20'] = processed_full.groupby('tic')['close'].rolling(window=20).std().reset_index(0, drop=True)
    
    # Fill NaN values
    processed_full = processed_full.bfill().ffill()
    
    return processed_full

def train_enhanced_model(train_data, model_name='enhanced_ppo'):
    """
    Train a single enhanced PPO model
    """
    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    # Environment
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    # Enhanced PPO configuration
    ppo_params = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 128,
    }
    
    # PPO agent
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo", model_kwargs=ppo_params)
    
    # Set up logger
    tmp_path = RESULTS_DIR + f'/{model_name}'
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger)
    
    trained_ppo = agent.train_model(
        model=model_ppo,
        tb_log_name=model_name,
        total_timesteps=80000
    )
    
    # Save model
    trained_ppo.save(TRAINED_MODEL_DIR + f'/{model_name}')
    
    return trained_ppo

def test_model(test_data, model_path, model_name):
    """
    Test a single model
    """
    # Load model
    model = PPO.load(model_path)
    
    # Prepare test environment
    stock_dimension = len(test_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    # Environment
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs)
    
    # Get predictions
    df_result, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
    
    return df_result, df_actions

def compare_models(test_data, model_paths, model_names):
    """
    Compare multiple models
    """
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"Testing {model_name}...")
        try:
            df_result, df_actions = test_model(test_data, model_path, model_name)
            results[model_name] = df_result
            
            print(f"=============={model_name} Results===========")
            perf_stats = backtest_stats(account_value=df_result)
            print(perf_stats)
            print()
            
        except FileNotFoundError:
            print(f"Model {model_path} not found, skipping...")
    
    return results

def plot_comparison(results, save_path="model_comparison.png"):
    """
    Plot comparison of different models
    """
    plt.rcParams["figure.figsize"] = (15, 10)
    
    if len(results) == 1:
        fig, ax = plt.subplots(1, 1)
        model_name, df_result = list(results.items())[0]
        df_result.plot(ax=ax, title=f'{model_name} Performance')
    else:
        fig, axes = plt.subplots(len(results), 1, figsize=(15, 5*len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, df_result) in enumerate(results.items()):
            df_result.plot(ax=axes[i], title=f'{model_name} Performance')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Enhanced Trading Model Comparison')
    parser.add_argument('--start_date', default='2018-01-01', help='Training start date')
    parser.add_argument('--end_date', default='2019-01-01', help='Training end date')
    parser.add_argument('--test_start_date', default='2019-01-01', help='Test start date')
    parser.add_argument('--test_end_date', default='2020-01-01', help='Test end date')
    parser.add_argument('--data_file', default='train_data.csv', help='Data file path')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test')
    
    args = parser.parse_args()
    
    # Create directories
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])
    
    if args.mode == 'train':
        print("Creating enhanced dataset...")
        enhanced_data = create_enhanced_data(args.data_file)
        
        print("Preparing training data...")
        train_data = data_split(enhanced_data, args.start_date, args.end_date)
        
        print("Training enhanced model...")
        enhanced_model = train_enhanced_model(train_data, 'enhanced_ppo')
        
        print("Training completed!")
        
    elif args.mode == 'test':
        print("Preparing test data...")
        enhanced_data = create_enhanced_data(args.data_file)
        test_data = data_split(enhanced_data, args.test_start_date, args.test_end_date)
        
        # Test different models
        model_paths = [
            TRAINED_MODEL_DIR + '/trained_ppo',  # Original baseline
            TRAINED_MODEL_DIR + '/enhanced_ppo',  # Enhanced model
        ]
        
        model_names = [
            'Baseline PPO',
            'Enhanced PPO'
        ]
        
        # Compare models
        results = compare_models(test_data, model_paths, model_names)
        
        # Plot comparison
        if results:
            plot_comparison(results, "enhanced_vs_baseline.png")
            
            # Save results
            for model_name, df_result in results.items():
                filename = f"{model_name.lower().replace(' ', '_')}_results.csv"
                df_result.to_csv(filename, index=False)
                print(f"Results saved to {filename}")
        
        print("Comparison completed!")

if __name__ == '__main__':
    main()
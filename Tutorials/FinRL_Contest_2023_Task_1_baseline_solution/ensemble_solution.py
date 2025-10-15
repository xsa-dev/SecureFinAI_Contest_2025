import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.plot import backtest_stats
from stable_baselines3.common.logger import configure

class SimpleEnsembleAgent:
    """
    Simple ensemble agent that combines multiple PPO models with different configurations
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def predict(self, observation, deterministic=True):
        """
        Get ensemble prediction using weighted voting
        """
        predictions = []
        
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            predictions.append(action)
        
        # Weighted voting
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weight = self.weights[i]
            weighted_predictions.append(pred * weight)
        
        # Majority vote
        final_prediction = np.sum(weighted_predictions, axis=0)
        final_prediction = np.argmax(final_prediction, axis=1)
        
        # Ensure the action is in the correct format for the environment
        if len(final_prediction.shape) == 1:
            final_prediction = final_prediction.reshape(-1, 1)
        
        return final_prediction, None

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

def train_multiple_models(train_data, num_models=3):
    """
    Train multiple PPO models with different configurations
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
    
    # Different PPO configurations
    ppo_configs = [
        {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "batch_size": 128,
        },
        {
            "n_steps": 1024,
            "ent_coef": 0.02,
            "learning_rate": 0.0005,
            "batch_size": 64,
        },
        {
            "n_steps": 4096,
            "ent_coef": 0.005,
            "learning_rate": 0.0001,
            "batch_size": 256,
        }
    ]
    
    models = []
    agent = DRLAgent(env=env_train)
    
    for i, config in enumerate(ppo_configs):
        print(f"Training model {i+1} with config: {config}")
        
        model_ppo = agent.get_model("ppo", model_kwargs=config)
        
        # Set up logger
        tmp_path = RESULTS_DIR + f'/ppo_ensemble_{i+1}'
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger)
        
        trained_ppo = agent.train_model(
            model=model_ppo,
            tb_log_name=f'ppo_ensemble_{i+1}',
            total_timesteps=80000
        )
        
        # Save model
        trained_ppo.save(TRAINED_MODEL_DIR + f'/trained_ppo_ensemble_{i+1}')
        models.append(trained_ppo)
    
    return models

def test_ensemble_models(test_data, model_paths, weights=None):
    """
    Test ensemble of models
    """
    # Load models
    models = []
    for path in model_paths:
        try:
            model = PPO.load(path)
            models.append(model)
        except FileNotFoundError:
            print(f"Model {path} not found, skipping...")
    
    if not models:
        raise ValueError("No models found to create ensemble")
    
    # Create ensemble
    ensemble_agent = SimpleEnsembleAgent(models, weights)
    
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
    df_result, df_actions = DRLAgent.DRL_prediction(model=ensemble_agent, environment=e_trade_gym)
    
    return df_result, df_actions

def main():
    parser = argparse.ArgumentParser(description='Ensemble Trading Agent')
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
        
        print("Training multiple models...")
        models = train_multiple_models(train_data, num_models=3)
        
        print("Training completed!")
        
    elif args.mode == 'test':
        print("Preparing test data...")
        enhanced_data = create_enhanced_data(args.data_file)
        test_data = data_split(enhanced_data, args.test_start_date, args.test_end_date)
        
        # Test with different ensemble configurations
        model_paths = [
            TRAINED_MODEL_DIR + '/trained_ppo_ensemble_1',
            TRAINED_MODEL_DIR + '/trained_ppo_ensemble_2',
            TRAINED_MODEL_DIR + '/trained_ppo_ensemble_3'
        ]
        
        # Test equal weights ensemble
        print("Testing equal weights ensemble...")
        df_result_equal, df_actions_equal = test_ensemble_models(
            test_data, model_paths, weights=[1.0, 1.0, 1.0]
        )
        
        print("==============Equal Weights Ensemble Results===========")
        perf_stats_equal = backtest_stats(account_value=df_result_equal)
        print(perf_stats_equal)
        
        # Test weighted ensemble (favor first model)
        print("Testing weighted ensemble...")
        df_result_weighted, df_actions_weighted = test_ensemble_models(
            test_data, model_paths, weights=[0.5, 0.3, 0.2]
        )
        
        print("==============Weighted Ensemble Results===========")
        perf_stats_weighted = backtest_stats(account_value=df_result_weighted)
        print(perf_stats_weighted)
        
        # Plot results
        plt.rcParams["figure.figsize"] = (15, 10)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        
        df_result_equal.plot(ax=ax1, title='Equal Weights Ensemble')
        df_result_weighted.plot(ax=ax2, title='Weighted Ensemble')
        
        plt.tight_layout()
        plt.savefig("ensemble_comparison.png")
        plt.show()
        
        # Save results
        df_result_equal.to_csv("ensemble_equal_results.csv", index=False)
        df_result_weighted.to_csv("ensemble_weighted_results.csv", index=False)
        
        print("Results saved to ensemble_*_results.csv and ensemble_comparison.png")

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.plot import backtest_stats

class EnsembleAgent:
    """
    Ensemble agent that combines multiple RL models using majority voting
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
        confidences = []
        
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            predictions.append(action)
            
            # Get confidence from model's action probabilities
            if hasattr(model, 'action_prob'):
                prob = model.action_prob(observation)
                confidence = np.max(prob)
            else:
                confidence = 1.0  # Default confidence
            confidences.append(confidence)
        
        # Weighted voting
        weighted_predictions = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            weight = self.weights[i] * conf
            weighted_predictions.append(pred * weight)
        
        # Majority vote
        final_prediction = np.sum(weighted_predictions, axis=0)
        final_prediction = np.argmax(final_prediction, axis=1)
        
        return final_prediction, None

class ImprovedTradingAgent:
    """
    Improved trading agent with ensemble methods and better feature engineering
    """
    def __init__(self, data_file='train_data.csv'):
        self.data_file = data_file
        self.models = []
        self.ensemble_agent = None
        
    def prepare_data(self, start_date, end_date):
        """
        Prepare and enhance data with additional features
        """
        processed_full = pd.read_csv(self.data_file)
        
        # Add additional technical indicators
        processed_full = self.add_technical_indicators(processed_full)
        
        # Split data
        trade_data = data_split(processed_full, start_date, end_date)
        
        return trade_data
    
    def add_technical_indicators(self, df):
        """
        Add additional technical indicators for better feature engineering
        """
        # Calculate additional moving averages
        for window in [5, 10, 20, 50]:
            df[f'close_{window}_sma'] = df.groupby('tic')['close'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'volume_{window}_sma'] = df.groupby('tic')['volume'].rolling(window=window).mean().reset_index(0, drop=True)
        
        # Price momentum indicators
        df['price_momentum_5'] = df.groupby('tic')['close'].pct_change(5)
        df['price_momentum_10'] = df.groupby('tic')['close'].pct_change(10)
        
        # Volume indicators
        df['volume_ratio'] = df['volume'] / df.groupby('tic')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
        
        # Volatility indicators
        df['volatility_20'] = df.groupby('tic')['close'].rolling(window=20).std().reset_index(0, drop=True)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def create_environment(self, trade_data, initial_amount=1000000):
        """
        Create trading environment with enhanced configuration
        """
        stock_dimension = len(trade_data.tic.unique())
        
        # Enhanced state space with additional features
        additional_features = 15  # New technical indicators
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension + additional_features*stock_dimension
        
        print(f"Stock Dimension: {stock_dimension}, Enhanced State Space: {state_space}")
        
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension
        
        env_kwargs = {
            "hmax": 100,
            "initial_amount": initial_amount,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        
        return StockTradingEnv(df=trade_data, **env_kwargs)
    
    def train_models(self, train_data, model_types=['PPO', 'DQN', 'A2C']):
        """
        Train multiple models for ensemble
        """
        env = self.create_environment(train_data)
        env_train, _ = env.get_sb_env()
        
        agent = DRLAgent(env=env_train)
        
        # PPO configuration
        ppo_params = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "batch_size": 128,
        }
        
        # DQN configuration
        dqn_params = {
            "learning_rate": 0.0001,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 1000,
        }
        
        # A2C configuration
        a2c_params = {
            "n_steps": 5,
            "learning_rate": 0.0007,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
        }
        
        models = []
        
        for model_type in model_types:
            print(f"Training {model_type} model...")
            
            if model_type == 'PPO':
                model = agent.get_model("ppo", model_kwargs=ppo_params)
            elif model_type == 'DQN':
                model = agent.get_model("dqn", model_kwargs=dqn_params)
            elif model_type == 'A2C':
                model = agent.get_model("a2c", model_kwargs=a2c_params)
            else:
                continue
            
            # Set up logger
            tmp_path = RESULTS_DIR + f'/{model_type.lower()}'
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            
            # Train model
            trained_model = agent.train_model(
                model=model,
                tb_log_name=model_type.lower(),
                total_timesteps=80000
            )
            
            # Save model
            trained_model.save(TRAINED_MODEL_DIR + f'/trained_{model_type.lower()}')
            models.append(trained_model)
            
        self.models = models
        return models
    
    def create_ensemble(self, weights=None):
        """
        Create ensemble agent from trained models
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_models first.")
        
        self.ensemble_agent = EnsembleAgent(self.models, weights)
        return self.ensemble_agent
    
    def test_ensemble(self, test_data):
        """
        Test ensemble agent on test data
        """
        if not self.ensemble_agent:
            raise ValueError("No ensemble agent created. Call create_ensemble first.")
        
        env = self.create_environment(test_data)
        
        # Get predictions from ensemble
        obs = env.reset()
        done = False
        actions = []
        
        while not done:
            action, _ = self.ensemble_agent.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, info = env.step(action)
        
        return env, actions

def main():
    parser = argparse.ArgumentParser(description='Improved Trading Agent with Ensemble Methods')
    parser.add_argument('--start_date', default='2018-01-01', help='Training start date')
    parser.add_argument('--end_date', default='2019-01-01', help='Training end date')
    parser.add_argument('--test_start_date', default='2019-01-01', help='Test start date')
    parser.add_argument('--test_end_date', default='2020-01-01', help='Test end date')
    parser.add_argument('--data_file', default='train_data.csv', help='Data file path')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test')
    
    args = parser.parse_args()
    
    # Create directories
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])
    
    # Initialize agent
    agent = ImprovedTradingAgent(args.data_file)
    
    if args.mode == 'train':
        # Prepare training data
        train_data = agent.prepare_data(args.start_date, args.end_date)
        
        # Train models
        models = agent.train_models(train_data, ['PPO', 'DQN', 'A2C'])
        
        # Create ensemble
        ensemble = agent.create_ensemble(weights=[0.4, 0.3, 0.3])  # Weighted ensemble
        print("Training completed!")
        
    elif args.mode == 'test':
        # Load trained models
        try:
            ppo_model = PPO.load(TRAINED_MODEL_DIR + '/trained_ppo')
            dqn_model = DQN.load(TRAINED_MODEL_DIR + '/trained_dqn')
            a2c_model = A2C.load(TRAINED_MODEL_DIR + '/trained_a2c')
            
            agent.models = [ppo_model, dqn_model, a2c_model]
            agent.create_ensemble(weights=[0.4, 0.3, 0.3])
            
            # Prepare test data
            test_data = agent.prepare_data(args.test_start_date, args.test_end_date)
            
            # Test ensemble
            env, actions = agent.test_ensemble(test_data)
            
            # Get backtest results
            df_result, df_actions = DRLAgent.DRL_prediction(model=agent.ensemble_agent, environment=env)
            
            print("==============Get Backtest Results===========")
            perf_stats_all = backtest_stats(account_value=df_result)
            print(perf_stats_all)
            
            # Plot results
            plt.rcParams["figure.figsize"] = (15, 5)
            plt.figure()
            df_result.plot()
            plt.title('Ensemble Trading Performance')
            plt.savefig("ensemble_plot.png")
            plt.show()
            
            # Save results
            df_result.to_csv("ensemble_results.csv", index=False)
            print("Results saved to ensemble_results.csv and ensemble_plot.png")
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please run training first with --mode train")

if __name__ == '__main__':
    main()
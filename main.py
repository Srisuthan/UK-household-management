from models.agent import Household
from models.model import HouseholdEnergyModel
from q_learning_agent import QLearningAgent
from energy_environment import EnergyEnvironment, calculate_realistic_human_baseline
import numpy as np
from graph_generator import generate_comparison_graphs

class EnergyModel:
    def __init__(self, num_households, season):
        self.num_households = num_households
        self.season = season
        self.household_model = HouseholdEnergyModel(num_households, season)
        self.q_learning_agent = QLearningAgent(state_size=[2, 2, 2], action_size=2)

    def train_agent(self, episodes=2000):
        env = EnergyEnvironment(num_rooms=self.num_households, season=self.season)

        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.q_learning_agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.q_learning_agent.learn(state, action, reward, next_state)
                state = next_state

    def test_agent_exploitation(self):
        env = EnergyEnvironment(num_rooms=self.num_households, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_cost = 0

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = self.q_learning_agent.choose_action(state)  # Exploit best known action
                state, reward, done, _ = env.step(action)
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                total_energy_used += energy_used
                total_cost += cost

        # Calculate averages
        avg_energy_used = total_energy_used / (90 * 24)
        avg_cost = total_cost / (90 * 24)

        return avg_energy_used, total_energy_used, avg_cost, total_cost

    def run(self):
        print(f"\nTraining agent for {self.season} season...")
        self.train_agent()
        print(f"Testing agent for {self.season} season...")

        # Test agent performance
        avg_electricity_usage_trained, total_electricity_usage_trained, avg_cost_trained, total_cost_trained = self.test_agent_exploitation()

        # Test random policy performance
        avg_electricity_usage_random, total_electricity_usage_random, avg_cost_random, total_cost_random = self.test_random_policy()

        # Print results
        print("\nResults with trained agent:")
        print(f"{self.season.capitalize()} - Avg Reward: {avg_electricity_usage_trained}\n"
              f"Avg Electricity Usage: {avg_electricity_usage_trained} kWh\n"
              f"Total Electricity Usage: {total_electricity_usage_trained} kWh\n"
              f"Avg Cost: £{avg_cost_trained:.2f}\n"
              f"Total Cost: £{total_cost_trained:.2f}\n")

        print("\nResults with random policy:")
        print(f"{self.season.capitalize()} - Avg Reward: {avg_electricity_usage_random}\n"
              f"Avg Electricity Usage: {avg_electricity_usage_random} kWh\n"
              f"Total Electricity Usage: {total_electricity_usage_random} kWh\n"
              f"Avg Cost: £{avg_cost_random:.2f}\n"
              f"Total Cost: £{total_cost_random:.2f}\n")

        print("\nPerformance improvement with trained agent:")
        print(f"{self.season.capitalize()} - Avg Electricity Usage Reduction (vs Random): {avg_electricity_usage_random - avg_electricity_usage_trained} kWh\n"
              f"Avg Cost Reduction (vs Random): £{avg_cost_random - avg_cost_trained:.2f}\n"
              f"Total Electricity Usage Reduction (vs Random): {total_electricity_usage_random - total_electricity_usage_trained} kWh\n"
              f"Total Cost Reduction (vs Random): £{total_cost_random - total_cost_trained:.2f}\n")
        
        # Generate and save comparison graphs
        generate_comparison_graphs(self.season, avg_electricity_usage_trained, total_electricity_usage_trained, avg_cost_trained, total_cost_trained,
                                    avg_electricity_usage_random, total_electricity_usage_random, avg_cost_random, total_cost_random)


    def test_random_policy(self):
        env = EnergyEnvironment(num_rooms=self.num_households, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_cost = 0

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = [np.random.choice([0, 1]) for _ in range(len(state))]
                state, reward, done, _ = env.step(action)
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                total_energy_used += energy_used
                total_cost += cost

        # Calculate averages
        avg_energy_used = total_energy_used / (90 * 24)
        avg_cost = total_cost / (90 * 24)

        return avg_energy_used, total_energy_used, avg_cost, total_cost


if __name__ == "__main__":
    num_households = 100
    seasons = ['winter', 'summer']

    for season in seasons:
        print(f"Running model for {season}...")
        model = EnergyModel(num_households, season=season)
        model.run()
        # household_data = model.collect_data()

        # for data in household_data:
        #     print(f"House Type: {data[0]}, Num People: {data[1]}, Electricity Usage: {data[2]} kWh, Gas Usage: {data[3]} kWh, Energy Saving: {data[4]}")

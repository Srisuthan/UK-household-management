import gym
from gym import spaces
import numpy as np

class EnergyEnvironment(gym.Env):
    def __init__(self, num_rooms=1, season='winter'):
        super(EnergyEnvironment, self).__init__()
        
        # Define action and observation space
        # Actions: [light, washing_machine, fridge]
        self.action_space = spaces.MultiDiscrete([2, 2, 2])  # Each can be 0 or 1
        
        # Observation: [light, washing_machine, fridge]
        self.observation_space = spaces.MultiDiscrete([2, 2, 2])
        
        self.num_rooms = num_rooms
        self.season = season
        
        # Current state and day/hour tracking
        self.current_hour = 0
        self.current_day = 0
        self.state = [0, 0, 1]  # Initial state, fridge is always on
        
        # Set seasonal energy prices and appliance usage
        self.set_seasonal_parameters()

    def set_seasonal_parameters(self):
        self.peak_price = 24.50  # pence per kWh
        self.off_peak_price = 22.36  # pence per kWh

        if self.season == 'winter':
            self.appliance_usage = {
                'washing_machine': 0.12,
                'fridge': 0.10,
                'lighting': 0.08,
                'heating': 0.30,  # Heating added for winter
            }
        elif self.season == 'summer':
            self.appliance_usage = {
                'washing_machine': 0.10,
                'fridge': 0.12,
                'lighting': 0.05,
                'cooling': 0.25,  # Cooling added for summer
            }

    def reset(self):
        self.state = [0, 0, 1]  # Fridge always on
        self.current_hour = 0
        self.current_day = 0
        return np.array(self.state)

    def step(self, action):
        light, washing_machine, fridge = action
        
        # Fridge is always on
        fridge = 1

        # Calculate energy used
        energy_used = (
            light * 1.5 * self.num_rooms * self.appliance_usage['lighting'] +
            washing_machine * 2 * self.appliance_usage['washing_machine'] +
            fridge * 1 * self.appliance_usage['fridge']
        )

        # Add heating or cooling energy usage
        if 'heating' in self.appliance_usage:
            energy_used += 2.5 * self.appliance_usage['heating']
        if 'cooling' in self.appliance_usage:
            energy_used += 2.5 * self.appliance_usage['cooling']

        # Determine if it's peak or off-peak time
        if 7 <= self.current_hour < 17 or 19 <= self.current_hour < 23:
            price = self.peak_price / 100  # Convert pence to pounds
        else:
            price = self.off_peak_price / 100  # Convert pence to pounds

        # Calculate reward (negative of cost)
        reward = -energy_used * price

        # Update the state
        self.state = [light, washing_machine, fridge]

        # Advance to the next hour
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        done = self.current_day >= 90  # Simulation runs for 90 days

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        print(f"Day: {self.current_day}, Hour: {self.current_hour}, State: {self.state}")

# Calculate realistic human baseline energy usage and cost for 90 days
def calculate_realistic_human_baseline(env):
    state = env.reset()
    total_energy_used = 0
    total_cost = 0

    for _ in range(90):  # Simulate for 90 days
        for hour in range(24):
            if hour == 7:  # Washing Machine ON from 7 AM to 8 AM
                action = [0, 1, 1]  # Only Washing Machine and Fridge ON
            elif 18 <= hour < 22:  # Lights ON from 6 PM to 10 PM
                action = [1, 0, 1]  # Only Lights and Fridge ON
            else:  # Other hours
                action = [0, 0, 1]  # Only Fridge ON

            state, reward, done, _ = env.step(action)
            energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
            cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
            total_energy_used += energy_used
            total_cost += cost

    avg_energy_used = total_energy_used / (90 * 24)
    avg_cost = total_cost / (90 * 24)
    
    return avg_energy_used, total_energy_used, avg_cost, total_cost

"""
City simulation module using reinforcement learning for NeuroUrban system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt

from src.config.settings import Config

class CityEnvironment:
    """City environment for reinforcement learning simulation."""
    
    def __init__(self, city_config: Dict):
        """
        Initialize city environment.
        
        Args:
            city_config: Configuration for the city
        """
        self.city_config = city_config
        
        # City state variables
        self.population = city_config.get("population", 1000000)
        self.area_km2 = city_config.get("area_km2", 500)
        
        # Infrastructure metrics
        self.traffic_flow = 0.5  # 0-1 scale
        self.energy_consumption = 0.5
        self.waste_generation = 0.5
        self.air_quality = 0.7
        self.water_quality = 0.8
        self.green_coverage = 0.3
        
        # Economic metrics
        self.gdp_per_capita = city_config.get("gdp_per_capita", 50000)
        self.employment_rate = 0.85
        self.cost_of_living = 0.6
        
        # Social metrics
        self.happiness_index = 0.7
        self.safety_index = 0.8
        self.education_quality = 0.75
        self.healthcare_quality = 0.8
        
        # Time step
        self.time_step = 0
        self.max_steps = 365  # One year simulation
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.time_step = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state of the city."""
        state = np.array([
            self.traffic_flow,
            self.energy_consumption,
            self.waste_generation,
            self.air_quality,
            self.water_quality,
            self.green_coverage,
            self.employment_rate,
            self.cost_of_living,
            self.happiness_index,
            self.safety_index,
            self.education_quality,
            self.healthcare_quality,
            self.time_step / self.max_steps  # Normalized time
        ])
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action vector representing policy decisions
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Action interpretation
        # action[0]: Transportation investment (0-1)
        # action[1]: Energy policy (0-1, higher = more renewable)
        # action[2]: Environmental policy (0-1, higher = more green)
        # action[3]: Economic policy (0-1, higher = more business-friendly)
        # action[4]: Social policy (0-1, higher = more social programs)
        
        # Apply actions and update state
        self._apply_transportation_policy(action[0])
        self._apply_energy_policy(action[1])
        self._apply_environmental_policy(action[2])
        self._apply_economic_policy(action[3])
        self._apply_social_policy(action[4])
        
        # Update time
        self.time_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.time_step >= self.max_steps
        
        # Additional info
        info = {
            "traffic_flow": self.traffic_flow,
            "air_quality": self.air_quality,
            "happiness_index": self.happiness_index,
            "sustainability_score": self._calculate_sustainability_score()
        }
        
        return self.get_state(), reward, done, info
    
    def _apply_transportation_policy(self, investment: float):
        """Apply transportation policy."""
        # Higher investment improves traffic flow
        improvement = investment * 0.1
        self.traffic_flow = max(0, min(1, self.traffic_flow + improvement - 0.02))  # Natural degradation
        
        # Side effects
        if investment > 0.7:  # High investment
            self.air_quality += 0.01  # Better public transport
            self.cost_of_living += 0.005  # Higher taxes
    
    def _apply_energy_policy(self, renewable_focus: float):
        """Apply energy policy."""
        # Higher renewable focus reduces consumption and improves air quality
        if renewable_focus > 0.5:
            self.energy_consumption = max(0.2, self.energy_consumption - 0.01)
            self.air_quality = min(1, self.air_quality + 0.02)
        else:
            self.energy_consumption = min(1, self.energy_consumption + 0.005)
            self.air_quality = max(0, self.air_quality - 0.01)
    
    def _apply_environmental_policy(self, green_focus: float):
        """Apply environmental policy."""
        # Higher green focus improves green coverage and air quality
        if green_focus > 0.6:
            self.green_coverage = min(1, self.green_coverage + 0.02)
            self.air_quality = min(1, self.air_quality + 0.015)
            self.happiness_index = min(1, self.happiness_index + 0.01)
        
        # Waste management
        waste_reduction = green_focus * 0.02
        self.waste_generation = max(0.1, self.waste_generation - waste_reduction + 0.005)
    
    def _apply_economic_policy(self, business_friendly: float):
        """Apply economic policy."""
        # Business-friendly policies affect employment and cost of living
        if business_friendly > 0.6:
            self.employment_rate = min(1, self.employment_rate + 0.01)
            self.gdp_per_capita *= 1.001
            self.cost_of_living = min(1, self.cost_of_living + 0.01)
        else:
            self.employment_rate = max(0.5, self.employment_rate - 0.005)
            self.cost_of_living = max(0.3, self.cost_of_living - 0.005)
    
    def _apply_social_policy(self, social_programs: float):
        """Apply social policy."""
        # Social programs affect happiness, education, and healthcare
        if social_programs > 0.5:
            self.happiness_index = min(1, self.happiness_index + 0.015)
            self.education_quality = min(1, self.education_quality + 0.01)
            self.healthcare_quality = min(1, self.healthcare_quality + 0.01)
            self.safety_index = min(1, self.safety_index + 0.005)
            self.cost_of_living = min(1, self.cost_of_living + 0.01)  # Higher taxes
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on city performance."""
        # Multi-objective reward function
        sustainability = self._calculate_sustainability_score()
        livability = self._calculate_livability_score()
        economic_health = self._calculate_economic_score()
        
        # Weighted combination
        reward = (
            0.4 * sustainability +
            0.4 * livability +
            0.2 * economic_health
        )
        
        # Penalties for extreme values
        if self.cost_of_living > 0.9:
            reward -= 0.2
        if self.unemployment_rate > 0.3:
            reward -= 0.3
        if self.air_quality < 0.3:
            reward -= 0.2
        
        return reward
    
    def _calculate_sustainability_score(self) -> float:
        """Calculate sustainability score."""
        return (
            self.air_quality * 0.3 +
            self.water_quality * 0.2 +
            self.green_coverage * 0.2 +
            (1 - self.energy_consumption) * 0.15 +
            (1 - self.waste_generation) * 0.15
        )
    
    def _calculate_livability_score(self) -> float:
        """Calculate livability score."""
        return (
            self.happiness_index * 0.25 +
            self.safety_index * 0.2 +
            self.education_quality * 0.2 +
            self.healthcare_quality * 0.2 +
            (1 - self.traffic_flow) * 0.15  # Lower traffic is better
        )
    
    def _calculate_economic_score(self) -> float:
        """Calculate economic score."""
        return (
            self.employment_rate * 0.4 +
            (1 - self.cost_of_living) * 0.3 +  # Lower cost is better
            min(1, self.gdp_per_capita / 100000) * 0.3  # Normalized GDP
        )

class SimpleRLAgent:
    """Simple reinforcement learning agent for city management."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        """
        Initialize RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for policy updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple policy network (linear)
        self.policy_weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.value_weights = np.random.normal(0, 0.1, state_dim)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from current policy."""
        # Linear policy
        action_logits = np.dot(state, self.policy_weights)
        
        # Apply sigmoid to get actions in [0, 1]
        actions = 1 / (1 + np.exp(-action_logits))
        
        # Add exploration noise
        noise = np.random.normal(0, 0.1, self.action_dim)
        actions = np.clip(actions + noise, 0, 1)
        
        return actions
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Store experience for learning."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def update_policy(self):
        """Update policy based on collected experience."""
        if len(self.states) == 0:
            return
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        
        # Simple policy gradient update
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            
            # Policy gradient
            predicted_action = 1 / (1 + np.exp(-np.dot(state, self.policy_weights)))
            action_error = action - predicted_action
            
            # Update weights
            self.policy_weights += self.learning_rate * np.outer(state, action_error * reward)
        
        # Clear experience
        self.states = []
        self.actions = []
        self.rewards = []

class CitySimulator:
    """Main city simulator using reinforcement learning."""
    
    def __init__(self, config: Config):
        """
        Initialize city simulator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Simulation components
        self.environment = None
        self.agent = None
        
        # Results storage
        self.simulation_results = {}
        
    def run_simulation(self, city_config: Optional[Dict] = None, episodes: int = None) -> Dict:
        """
        Run city simulation.
        
        Args:
            city_config: Configuration for the city to simulate
            episodes: Number of episodes to run
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info("Starting city simulation...")
        
        # Default configuration
        if city_config is None:
            city_config = {
                "population": 1000000,
                "area_km2": 500,
                "gdp_per_capita": 50000
            }
        
        if episodes is None:
            episodes = self.config.model.rl_episodes
        
        # Initialize environment and agent
        self.environment = CityEnvironment(city_config)
        state_dim = len(self.environment.get_state())
        action_dim = 5  # 5 policy dimensions
        
        self.agent = SimpleRLAgent(state_dim, action_dim, self.config.model.rl_learning_rate)
        
        # Run simulation episodes
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(episodes):
            episode_reward, episode_metric = self._run_episode()
            episode_rewards.append(episode_reward)
            episode_metrics.append(episode_metric)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        # Compile results
        self.simulation_results = {
            "city_config": city_config,
            "episodes": episodes,
            "episode_rewards": episode_rewards,
            "episode_metrics": episode_metrics,
            "final_metrics": episode_metrics[-1] if episode_metrics else {},
            "average_reward": np.mean(episode_rewards),
            "best_episode": int(np.argmax(episode_rewards)),
            "best_reward": float(np.max(episode_rewards))
        }
        
        # Save results
        self._save_simulation_results()
        
        self.logger.info("✅ City simulation completed")
        return self.simulation_results
    
    def _run_episode(self) -> Tuple[float, Dict]:
        """Run a single simulation episode."""
        state = self.environment.reset()
        total_reward = 0
        episode_metrics = []
        
        for step in range(self.environment.max_steps):
            # Get action from agent
            action = self.agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward)
            
            # Update totals
            total_reward += reward
            episode_metrics.append(info.copy())
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Update agent policy
        self.agent.update_policy()
        
        # Calculate final metrics
        final_metrics = {
            "total_reward": total_reward,
            "avg_sustainability": np.mean([m["sustainability_score"] for m in episode_metrics]),
            "avg_happiness": np.mean([m["happiness_index"] for m in episode_metrics]),
            "avg_air_quality": np.mean([m["air_quality"] for m in episode_metrics]),
            "avg_traffic_flow": np.mean([m["traffic_flow"] for m in episode_metrics])
        }
        
        return total_reward, final_metrics
    
    def _save_simulation_results(self):
        """Save simulation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.config.get_output_path(f"simulation_results_{timestamp}.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_json_serializable(self.simulation_results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"💾 Simulation results saved to: {results_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def generate_simulation_report(self) -> str:
        """Generate a text report of simulation results."""
        if not self.simulation_results:
            return "No simulation results available."
        
        report = []
        report.append("🏙️ NEUROURBAN CITY SIMULATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic info
        config = self.simulation_results["city_config"]
        report.append(f"Population: {config['population']:,}")
        report.append(f"Area: {config['area_km2']} km²")
        report.append(f"GDP per capita: ${config['gdp_per_capita']:,}")
        report.append("")
        
        # Performance metrics
        final_metrics = self.simulation_results["final_metrics"]
        report.append("📊 FINAL PERFORMANCE METRICS:")
        report.append(f"Average Reward: {self.simulation_results['average_reward']:.3f}")
        report.append(f"Best Episode Reward: {self.simulation_results['best_reward']:.3f}")
        report.append(f"Sustainability Score: {final_metrics.get('avg_sustainability', 0):.3f}")
        report.append(f"Happiness Index: {final_metrics.get('avg_happiness', 0):.3f}")
        report.append(f"Air Quality: {final_metrics.get('avg_air_quality', 0):.3f}")
        report.append(f"Traffic Flow: {final_metrics.get('avg_traffic_flow', 0):.3f}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if final_metrics.get('avg_sustainability', 0) < 0.7:
            report.append("- Focus on environmental policies to improve sustainability")
        if final_metrics.get('avg_happiness', 0) < 0.7:
            report.append("- Increase social programs to improve citizen happiness")
        if final_metrics.get('avg_air_quality', 0) < 0.7:
            report.append("- Implement stricter environmental regulations")
        if final_metrics.get('avg_traffic_flow', 0) > 0.7:
            report.append("- Invest in public transportation infrastructure")
        
        return "\n".join(report)

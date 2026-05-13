import numpy as np
import quaternion
import habitat
from habitat.tasks.nav.nav import EpisodicGPSSensor
from habitat.config.default_structured_configs import GPSSensorConfig

# Mock a simple simulator to bypass loading datasets
class MockSim:
    def get_agent_state(self):
        class State:
            # position in habitat is (x, y_up, z)
            position = np.array([1.5, 3.0, -2.5]) 
            rotation = np.quaternion(1, 0, 0, 0)
        return State()

sim = MockSim()

print("--- Testing Habitat GPSSensor ---")

# 1. Test Default 2D GPS (What glosm-nav currently uses)
config_2d = GPSSensorConfig()
sensor_2d = EpisodicGPSSensor(sim=sim, config=config_2d)
print(f"Default 2D GPS Observation Space: {sensor_2d.observation_space}")
print(f"Default 2D GPS Reading: {sensor_2d.get_observation(observations={}, sim=sim, task=None, episode=habitat.core.dataset.Episode(start_position=[0,0,0], start_rotation=[0,0,0,1], episode_id='0', scene_id='0'))}")

# 2. Test 3D GPS (What we want to implement)
config_3d = GPSSensorConfig(dimensionality=3)
sensor_3d = EpisodicGPSSensor(sim=sim, config=config_3d)
print(f"\n3D GPS Observation Space: {sensor_3d.observation_space}")
print(f"3D GPS Reading: {sensor_3d.get_observation(observations={}, sim=sim, task=None, episode=habitat.core.dataset.Episode(start_position=[0,0,0], start_rotation=[0,0,0,1], episode_id='0', scene_id='0'))}")

import hydra
from omegaconf import DictConfig, OmegaConf
from src import environment
from src import utils
import lightning as L

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

log = utils.get_pylogger(__name__)

TEST_INDEX_START = 4380
TEST_INDEX_END = 8500
BATTERY_CAPACITY = 400
BATTERY_POWER = 100
NUM_FORECAST_STEPS = 8
RESULT_PATH = "rl_example/"


def execute(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    load, price, generation = utils.read_data("./data/preprocessed")
    load_train = load[:TEST_INDEX_START]
    price_train = price[:TEST_INDEX_START]
    generation_train = generation[:TEST_INDEX_START]

    sim = environment.BuildingSimulation(
        electricity_load_profile=load_train,
        solar_generation_profile=generation_train,
        electricity_price=price_train,
        max_battery_charge_per_timestep=BATTERY_POWER,
        battery_capacity=BATTERY_CAPACITY,
    )

    env = environment.Environment(
        sim,
        num_forecasting_steps=NUM_FORECAST_STEPS,
        max_timesteps=len(load_train) - NUM_FORECAST_STEPS,
    )
    env = utils.ObservationWrapper(env, NUM_FORECAST_STEPS)
    initial_obs, info = env.reset()
    print(initial_obs)

    env = Monitor(env, filename=RESULT_PATH)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Train :-)
    model = SAC("MlpPolicy", env, verbose=1, gamma=0.95)
    model.learn(total_timesteps=200_000)
    # Store the trained Model and environment stats (which are needed as we are standardizing the observations and
    # reward using VecNormalize())
    model.save(RESULT_PATH + "model")
    env.save(RESULT_PATH + "env.pkl")


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == "__main__":
    main()

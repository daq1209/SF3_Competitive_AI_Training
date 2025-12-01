import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack 
from gymnasium.wrappers import FlattenObservation # BẮT BUỘC để fix lỗi Nested Dict
import os
import numpy as np 

# ====================================================================
# B1. KHAI BÁO BIẾN MÔI TRƯỜNG & KEY
# ====================================================================

os.environ["DIAMBRA_APP_KEY"] = "34658be6b3f04b6bbf76ecf40bf575c5"
os.environ["DIAMBRA_ENVS"] = "127.0.0.1:50051" 


def create_diambra_env():
    """Hàm tạo môi trường, trả về định dạng đơn giản nhất cho PPO."""
    env_settings = EnvironmentSettings()
    env_settings.difficulty = 3
    env_settings.characters = "Ryu" 
    
    wrappers_settings = WrappersSettings()
    wrappers_settings.hardcore = True
    wrappers_settings.frame_shape = (128, 128, 1) 
    wrappers_settings.stack_frames = 1 
    wrappers_settings.add_last_action = False 
    wrappers_settings.compact_observation = False 

    # CHUYỂN SANG CHẾ ĐỘ ẨN HÌNH (TỐC ĐỘ CAO)
    env = diambra.arena.make(
        "sfiii3n", 
        env_settings, 
        wrappers_settings, 
        render_mode="rgb_array" # <-- CHẾ ĐỘ TỐI ƯU CHO TRAINING
    )
    
    # ÁP DỤNG FLATTEN (Bắt buộc)
    env = FlattenObservation(env) 
    return env


def main():
    print("--- 1. TẠO MÔI TRƯỜNG GỐC VÀ DUMMY VEC ---")
    
    env = DummyVecEnv([create_diambra_env])

    print("--- 2. STACK KHUNG HÌNH CUỐI CÙNG ---")
    env = VecFrameStack(env, n_stack=4, channels_order="last") 
    
    print("--- 3. KHỞI TẠO AI (PPO) ---")
    # KHÔI PHỤC CÁC THÔNG SỐ BAN ĐẦU (TỐC ĐỘ CAO)
    model = PPO(
        "MlpPolicy", # Policy chuẩn cho dữ liệu đã Flatten
        env, 
        verbose=1, 
        tensorboard_log="./logs/", 
        learning_rate=0.0003,
        n_steps=2048,          # Khôi phục: GOM DATA LỚN HƠN
        batch_size=256         # Khôi phục: TÍNH TOÁN NHANH HƠN
    )

    print("--- 4. BẮT ĐẦU TRAIN (TỐC ĐỘ TỐI ƯU) ---")
    # Tổng cộng 100,000 bước train
    model.learn(total_timesteps=100000)

    print("--- 5. TRAIN XONG! LƯU MODEL ---")
    model.save("sf3_ryu_model")
    env.close()

if __name__ == "__main__":
    main()
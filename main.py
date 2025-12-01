import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings # <--- Nhập thêm 2 món này
import os
import time

# 1. Cấu hình Key & ROM
os.environ["DIAMBRA_APP_KEY"] = "34658be6b3f04b6bbf76ecf40bf575c5"
# Lưu ý: Nếu chạy thủ công bằng Docker thì dòng ROM này có thể thừa nhưng cứ để cho chắc
os.environ["DIAMBRA_ROM_PATH"] = os.path.join(os.getcwd(), "roms")

def main():
    print("--- ĐANG CẤU HÌNH GAME ---")

    # 2. Cài đặt thông số Game (Environment Settings)
    # Thay vì dùng dict {}, ta dùng class EnvironmentSettings()
    env_settings = EnvironmentSettings()
    env_settings.difficulty = 3      # Độ khó
    env_settings.characters = "Ryu"  # Nhân vật

    # 3. Cài đặt hình ảnh/xử lý (Wrappers Settings)
    wrappers_settings = WrappersSettings()
    wrappers_settings.hardcore = True # Tắt hỗ trợ
    wrappers_settings.frame_shape = (128, 128, 1) # Ảnh xám 128x128
    
    print("--- ĐANG KẾT NỐI VỚI SERVER ---")
    # Lệnh tạo môi trường chuẩn
    env = diambra.arena.make("sfiii3n", env_settings, wrappers_settings, render_mode="human")
    
    observation, info = env.reset(seed=42)
    print("--- GAME BẮT ĐẦU (TỐC ĐỘ CHUẨN 60 FPS) ---")

    while True:
        env.render()
        
        # Hãm phanh để mắt người nhìn kịp
        time.sleep(0.016) 

        # Random hành động
        actions = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
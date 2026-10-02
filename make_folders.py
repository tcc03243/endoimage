import os

base_dir = "./dataset"
os.makedirs(base_dir, exist_ok=True)

# 1. 자연치 대분류 (16개)
nat_types = ["상악절치", "상악견치", "상악소구치", "상악대구치", "하악절치", "하악견치", "하악소구치", "하악대구치"]
# 2. SRT 인공치 (32개)
srt_types = [f"{i}{j}" for i in range(1, 5) for j in range(1, 9)]

dirs = ["협설", "근원"]

count = 0
for t in nat_types:
    for d in dirs:
        os.makedirs(os.path.join(base_dir, f"자연치_{t}_{d}"), exist_ok=True)
        count += 1

for t in srt_types:
    for d in dirs:
        os.makedirs(os.path.join(base_dir, f"SRT_{t}_{d}"), exist_ok=True)
        count += 1

print(f"완료! '{base_dir}' 안에 총 {count}개의 정답 폴더가 생성되었습니다.")
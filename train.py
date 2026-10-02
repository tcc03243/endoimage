import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# --- 1. 기본 설정 ---
data_dir = './dataset'
batch_size = 16
num_epochs = 15         
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 스마트 데이터셋 클래스 (폴더 삭제 안 함!) ---
# PyTorch의 기본 ImageFolder를 개조하여 빈 폴더를 무시하도록 만듭니다.
class SafeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # 1. 모든 폴더 목록을 가져옵니다.
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        valid_classes = []
        for class_name in classes:
            folder_path = os.path.join(directory, class_name)
            # 2. 폴더 안에 사진(.jpg, .png 등)이 있는지 검사합니다.
            has_image = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(folder_path))
            if has_image:
                valid_classes.append(class_name) # 사진이 있는 폴더만 합격!
        
        if not valid_classes:
            raise FileNotFoundError(f"🚨 '{directory}' 안에 이미지가 들어있는 폴더가 하나도 없습니다.")
            
        # 3. 합격한 폴더들에게만 정답 번호(Index)를 부여합니다. (빈 폴더는 디스크에 그대로 남음)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        return valid_classes, class_to_idx

# --- 3. 데이터 부풀리기 (Data Augmentation) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),                
    transforms.RandomRotation(degrees=15),                 
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. 데이터 불러오기 ---
# 기본 ImageFolder 대신 방금 만든 SafeImageFolder를 사용합니다.
dataset = SafeImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.classes)

print(f"✅ 인식된 치아 종류 수: {num_classes}개 (빈 폴더는 삭제하지 않고 무시했습니다)")
# 학습에 사용될 실제 폴더 목록을 짧게 보여줍니다.
print(f"📌 학습 대상: {dataset.classes[:5]} ... 등등")

# --- 5. AI 모델 뇌 구조 가져오기 ---
model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. 본격적인 학습 시작 ---
print(f"\n--- 🚀 {device} 환경에서 학습을 시작합니다 ---")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{num_epochs} 완료 | 오차(Loss): {running_loss/len(dataloader):.4f}")

# --- 7. 똑똑해진 뇌(가중치) 저장 ---
torch.save(model.state_dict(), 'dental_model.pth')
print("🎉 학습 완료! 'dental_model.pth' 파일이 성공적으로 생성되었습니다.")
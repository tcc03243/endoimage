import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import timm

# ---------------------------------------------------------
# 1. 커스텀 데이터셋 (빈 폴더 무시 기능 추가!)
# ---------------------------------------------------------
class DentalMultiTaskDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        
        # --- 수정된 부분: 사진이 들어있는 폴더만 정답지로 씁니다 ---
        all_classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.classes = []
        for cls_name in all_classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(cls_dir)):
                self.classes.append(cls_name)
        # -----------------------------------------------------------
                
        self.image_paths = []
        self.labels = []
        
        for idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.base_transform(image)
        class_label = self.labels[idx]

        rot_label = 0  # 0: 0도, 1: 반시계90도, 2: 180도, 3: 반시계270도

        if self.is_train:
            rot_label = random.choice([0, 1, 2, 3])
            if rot_label > 0:
                img_tensor = torch.rot90(img_tensor, k=rot_label, dims=[1, 2])

        return img_tensor, class_label, rot_label

# ---------------------------------------------------------
# 2. 투-헤드(Two-Head) 모델
# ---------------------------------------------------------
class MultiHeadDentalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadDentalModel, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        self.fc_class = nn.Linear(num_features, num_classes)
        self.fc_rot = nn.Linear(num_features, 4)           

    def forward(self, x):
        features = self.backbone(x)
        out_class = self.fc_class(features)
        out_rot = self.fc_rot(features)
        return out_class, out_rot

# ---------------------------------------------------------
# 3. 메인 학습 코드
# ---------------------------------------------------------
def train_model():
    dataset_path = "./dataset"  
    batch_size = 16
    epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")

    train_dataset = DentalMultiTaskDataset(dataset_path, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(train_dataset.classes)
    
    # 여기서 "22개"라고 떠야 정상입니다!
    print(f"인식할 치아 종류: {num_classes}개")

    model = MultiHeadDentalModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("🚀 V2.0 (회전 인식 특화) 딥러닝 학습을 시작합니다!")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels_class, labels_rot in train_loader:
            images = images.to(device)
            labels_class = labels_class.to(device)
            labels_rot = labels_rot.to(device)
            
            optimizer.zero_grad()
            
            pred_class, pred_rot = model(images)
            
            loss_class = criterion(pred_class, labels_class)
            loss_rot = criterion(pred_rot, labels_rot)
            
            total_loss = loss_class + loss_rot
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "dental_model_v2.pth")
    print("🎉 학습 완료! 'dental_model_v2.pth' 파일이 생성되었습니다.")

if __name__ == "__main__":
    train_model()
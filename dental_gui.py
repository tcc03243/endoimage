import os
import sys
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import timm

# ==========================================
# 1. 리소스 경로 설정 (PyInstaller 단일 파일 지원)
# ==========================================
def resource_path(relative_path):
    """ PyInstaller로 빌드 시 임시 폴더(MEIPASS)에서 리소스를 찾고, 일반 실행 시 현재 폴더에서 찾음 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# 2. 딥러닝 모델 로드 및 추론 클래스
# ==========================================
class DentalAIModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        try:
            model_path = resource_path("dental_model.pth")
            # TODO: 실제 사용하신 timm 모델 아키텍처와 클래스 수(num_classes)에 맞게 수정하세요.
            self.model = timm.create_model('resnet18', pretrained=False, num_classes=10)
            
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print("모델 로드 성공")
            else:
                print(f"경고: {model_path} 가중치 파일이 없습니다. 더미 결과를 반환합니다.")
        except Exception as e:
            print(f"모델 로딩 에러: {e}")

    def predict(self, image_path, candidates=""):
        """
        AI 추론 부분: 실제 모델의 출력에 따라 치아종류, 번호, 방향을 반환하도록 수정하세요.
        현재는 예시(Dummy) 데이터를 반환합니다.
        """
        if self.model and os.path.exists(resource_path("dental_model.pth")):
            # 실제 추론 로직 (주석 해제 후 구현)
            # image = Image.open(image_path).convert('RGB')
            # input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            # with torch.no_grad():
            #     output = self.model(input_tensor)
            #     # output 후처리 로직...
            pass 
        
        # 임시 반환값 (치아종류, 치아번호, 촬영방향)
        return "자연치", "14", "협설측"

# ==========================================
# 3. GUI 애플리케이션 클래스
# ==========================================
class DentalAutoSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("치과 방사선 사진 자동 정렬기 (AI 기반)")
        self.root.geometry("1200x800")
        
        self.ai = DentalAIModel()
        self.image_data = [] # 모든 이미지의 메타데이터 보관
        self.folder_path = ""
        
        self.setup_ui()

    def setup_ui(self):
        # 상단 설정 프레임
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill="x", side="top")

        tk.Label(top_frame, text="인원당 사진 수:").pack(side="left")
        self.per_user_var = tk.IntVar(value=10)
        tk.Spinbox(top_frame, from_=1, to=50, textvariable=self.per_user_var, width=5).pack(side="left", padx=5)

        tk.Label(top_frame, text="후보 치아 번호 (예: 11,24,36):").pack(side="left", padx=(15,0))
        self.candidate_var = tk.StringVar()
        tk.Entry(top_frame, textvariable=self.candidate_var, width=20).pack(side="left", padx=5)

        tk.Button(top_frame, text="폴더 열기", command=self.load_folder, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Button(top_frame, text="전체 저장 및 파일 변환", command=self.save_all, bg="#008CBA", fg="white", font=("Arial", 10, "bold")).pack(side="right", padx=10)

        # 메인 캔버스 프레임 (가로, 세로 스크롤)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray90")
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="gray90")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

    def load_folder(self):
        self.folder_path = filedialog.askdirectory(title="방사선 사진이 있는 폴더를 선택하세요")
        if not self.folder_path:
            return

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.image_data.clear()

        valid_ext = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_ext)]
        files.sort()

        if not files:
            messagebox.showinfo("알림", "선택한 폴더에 이미지 파일이 없습니다.")
            return

        per_user = self.per_user_var.get()
        candidates = self.candidate_var.get()

        users_files = [files[i:i + per_user] for i in range(0, len(files), per_user)]

        # 각 사용자별로 Row 생성
        for user_idx, user_files in enumerate(users_files):
            row_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove", pady=5)
            row_frame.pack(fill="x", pady=5, padx=5, anchor="w")

            # 좌측: 학생 정보 입력칸
            info_frame = tk.Frame(row_frame, width=150)
            info_frame.pack(side="left", fill="y", padx=10)
            
            tk.Label(info_frame, text=f"학생 {user_idx+1}", font=("Arial", 10, "bold")).pack(pady=(10, 0))
            tk.Label(info_frame, text="교번:").pack()
            id_var = tk.StringVar()
            tk.Entry(info_frame, textvariable=id_var, width=10).pack()
            
            tk.Label(info_frame, text="이름:").pack()
            name_var = tk.StringVar()
            tk.Entry(info_frame, textvariable=name_var, width=10).pack()

            # 우측: 이미지 나열
            images_frame = tk.Frame(row_frame)
            images_frame.pack(side="left", fill="x", expand=True)

            for file in user_files:
                img_path = os.path.join(self.folder_path, file)
                self.create_image_panel(images_frame, img_path, id_var, name_var, candidates)

    def create_image_panel(self, parent, img_path, id_var, name_var, candidates):
        panel = tk.Frame(parent, padx=5, pady=5)
        panel.pack(side="left", anchor="n")

        # 1. AI 추론
        tooth_type, tooth_num, view_dir = self.ai.predict(img_path, candidates)
        
        # 2. 해부학적 기준 자동 회전 (상악 10, 20번대는 180도 회전)
        rotation_angle = 180 if str(tooth_num).startswith(('1', '2')) else 0
        is_flipped = False

        # 메타데이터 저장용 딕셔너리
        img_meta = {
            "path": img_path,
            "id_var": id_var,
            "name_var": name_var,
            "tooth_type": tk.StringVar(value=tooth_type),
            "tooth_num": tk.StringVar(value=tooth_num),
            "view_dir": tk.StringVar(value=view_dir),
            "rotation": rotation_angle,
            "flipped": is_flipped,
            "img_label": tk.Label(panel)
        }
        self.image_data.append(img_meta)

        # 이미지 썸네일 라벨
        img_meta["img_label"].pack()
        self.update_thumbnail(img_meta)

        # 컨트롤 버튼 및 입력칸
        ctrl_frame = tk.Frame(panel)
        ctrl_frame.pack(fill="x", pady=2)
        
        tk.Entry(ctrl_frame, textvariable=img_meta["tooth_type"], width=5).pack(side="left")
        tk.Entry(ctrl_frame, textvariable=img_meta["tooth_num"], width=3).pack(side="left")
        tk.Entry(ctrl_frame, textvariable=img_meta["view_dir"], width=5).pack(side="left")

        btn_frame = tk.Frame(panel)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="↻ 180도", command=lambda m=img_meta: self.rotate_image(m)).pack(side="left", expand=True)
        tk.Button(btn_frame, text="↔ 좌우반전", command=lambda m=img_meta: self.flip_image(m)).pack(side="left", expand=True)

    def update_thumbnail(self, img_meta):
        img = Image.open(img_meta["path"])
        if img_meta["flipped"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if img_meta["rotation"] == 180:
            img = img.rotate(180)
            
        img.thumbnail((150, 150))
        photo = ImageTk.PhotoImage(img)
        img_meta["img_label"].config(image=photo)
        img_meta["img_label"].image = photo # 가비지 컬렉션 방지

    def rotate_image(self, img_meta):
        img_meta["rotation"] = (img_meta["rotation"] + 180) % 360
        self.update_thumbnail(img_meta)

    def flip_image(self, img_meta):
        img_meta["flipped"] = not img_meta["flipped"]
        self.update_thumbnail(img_meta)

    def save_all(self):
        if not self.image_data:
            messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
            return

        result_dir = os.path.join(self.folder_path, "결과물")
        os.makedirs(result_dir, exist_ok=True)

        success_count = 0
        for meta in self.image_data:
            student_id = meta["id_var"].get().strip() or "미상"
            student_name = meta["name_var"].get().strip() or "이름없음"
            t_type = meta["tooth_type"].get().strip()
            t_num = meta["tooth_num"].get().strip()
            v_dir = meta["view_dir"].get().strip()

            # 파일명 규칙: 교번_이름_치아종류+치아번호_방향.jpg
            base_name = f"{student_id}_{student_name}_{t_type}{t_num}_{v_dir}"
            ext = os.path.splitext(meta["path"])[1]
            
            new_filename = f"{base_name}{ext}"
            new_filepath = os.path.join(result_dir, new_filename)

            # 중복 방지 로직
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{base_name}({counter}){ext}"
                new_filepath = os.path.join(result_dir, new_filename)
                counter += 1

            # 원본 보존 및 편집 반영 복사 저장
            img = Image.open(meta["path"])
            if meta["flipped"]:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if meta["rotation"] == 180:
                img = img.rotate(180)
            
            # 메타데이터 보존하며 고화질 저장
            img.save(new_filepath, quality=100)
            success_count += 1

        messagebox.showinfo("완료", f"총 {success_count}개의 파일이 '{result_dir}' 폴더에 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAutoSorterApp(root)
    root.mainloop()
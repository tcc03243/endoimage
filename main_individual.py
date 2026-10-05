import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import openpyxl
import time

def get_executable_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))

# ---------------------------------------------------------
# [V2.0] 투-헤드(Two-Head) 모델 
# ---------------------------------------------------------
class MultiHeadDentalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadDentalModel, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        
        self.fc_class = nn.Linear(num_features, num_classes)
        self.fc_rot = nn.Linear(num_features, 4)           

    def forward(self, x):
        features = self.backbone(x)
        out_class = self.fc_class(features)
        out_rot = self.fc_rot(features)
        return out_class, out_rot

class DentalAIModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = [] 
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_class_names()
        self.load_model()

    def load_class_names(self):
        base_dir = get_executable_dir()
        class_txt_path = os.path.join(base_dir, "class_names.txt")
        dataset_dir = os.path.join(base_dir, "dataset")

        if os.path.exists(class_txt_path):
            with open(class_txt_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        elif os.path.exists(dataset_dir):
            classes = sorted(entry.name for entry in os.scandir(dataset_dir) if entry.is_dir())
            valid_classes = []
            for class_name in classes:
                folder_path = os.path.join(dataset_dir, class_name)
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(folder_path)):
                    valid_classes.append(class_name)
            
            self.class_names = valid_classes
            with open(class_txt_path, 'w', encoding='utf-8') as f:
                for c in valid_classes:
                    f.write(f"{c}\n")

    def load_model(self):
        model_path = os.path.join(get_executable_dir(), "dental_model_v2.pth")
        try:
            if os.path.exists(model_path):
                self.model = MultiHeadDentalModel(len(self.class_names))
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ V2.0 자동 정렬 모델 로딩 성공!")
            else:
                print("경고: dental_model_v2.pth 가중치 파일이 없습니다.")
        except Exception as e:
            print(f"모델 로딩 에러: {e}")

    def predict(self, image_path, candidates):
        if self.model is None or not self.class_names:
            return "미상", "미상", "미상", 0
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = ImageOps.exif_transpose(image)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out_class, out_rot = self.model(input_tensor) 
                
                if candidates:
                    valid_indices = []
                    for i, name in enumerate(self.class_names):
                        if any(name.startswith(cand) for cand in candidates):
                            valid_indices.append(i)
                    if valid_indices:
                        mask = torch.ones_like(out_class, dtype=torch.bool)
                        mask[0, valid_indices] = False
                        out_class[mask] = -float('inf')
                
                _, predicted_class = torch.max(out_class, 1)
                _, predicted_rot = torch.max(out_rot, 1)
                
                idx = predicted_class.item()
                rot_label = predicted_rot.item()
                
            predicted_label = self.class_names[idx]
            parts = predicted_label.split('_')
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2], rot_label
            return predicted_label, "", "", rot_label
            
        except Exception as e:
            print(f"추론 중 에러 발생: {e}")
            return "에러", "에러", "에러", 0


class DentalAutoSorterApp_Individual:
    def __init__(self, root):
        self.root = root
        self.root.title("치과 방사선 사진 자동 정렬기 V2.1 (개인용)")
        self.root.geometry("1400x900")
        
        self.ai = DentalAIModel()
        self.image_data = []
        self.folder_path = ""
        self.naming_rules = {} 
        self.reverse_rules = {} 
        
        # 💡 [V2.1 개별용 엑셀 데이터 변수 추가]
        self.student_list = []
        self.student_mapping = {}
        
        self.id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        
        self.load_naming_rules() 
        self.load_excel_data()
        self.setup_ui()

    def load_naming_rules(self):
        base_dir = get_executable_dir()
        rule_path = os.path.join(base_dir, "naming_rules.txt")
        default_rules = {
            "SRT": "srt", "자연치": "nt",
            "상악절치": "상절", "상악견치": "상견", "상악소구치": "상소", 
            "상악제1대구치": "상대(1)", "상악제2대구치": "상대(2)",
            "하악절치": "하절", "하악견치": "하견", "하악소구치": "하소", 
            "하악제1대구치": "하대(1)", "하악제2대구치": "하대(2)",
            "근원": "근원심"
        }
        if not os.path.exists(rule_path):
            with open(rule_path, 'w', encoding='utf-8') as f:
                f.write("# 파일명 변환 규칙 설정 파일입니다.\n")
                for k, v in default_rules.items():
                    f.write(f"{k}={v}\n")
            self.naming_rules = default_rules
        else:
            with open(rule_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        self.naming_rules[k.strip()] = v.strip()
        self.reverse_rules = {v: k for k, v in self.naming_rules.items()}

    # 💡 [추가] 엑셀에서 전체 명단 불러오기
    def load_excel_data(self):
        base_dir = get_executable_dir()
        excel_path = os.path.join(base_dir, "students.xlsx")
        if not os.path.exists(excel_path):
            messagebox.showwarning("경고", f"'students.xlsx'가 없습니다.")
            return

        wb = openpyxl.load_workbook(excel_path, data_only=True)
        ws = wb.active
        
        for row in ws.iter_rows(min_row=1, values_only=True):
            if not row[0]: continue
            group = str(row[0])
            student_id = str(row[1]) if row[1] else ""
            name = str(row[2]) if row[2] else ""
            
            if name:
                display_text = f"[{group}] {student_id} - {name}" if student_id else f"[{group}] {name}"
                self.student_list.append(display_text)
                self.student_mapping[display_text] = (student_id, name)

    def setup_ui(self):
        self.setup_candidate_ui()

        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill="x", side="top")

        # 1️⃣ 왼쪽 블록 (개인 정보 입력: 엑셀 명단에서 콤보박스로 선택)
        left_section = tk.Frame(top_frame)
        left_section.pack(side="left", anchor="nw")

        tk.Label(left_section, text="본인 선택:", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5, pady=15, sticky="e")
        self.student_combo = ttk.Combobox(left_section, values=self.student_list, state="readonly", width=25, font=("Arial", 11))
        self.student_combo.grid(row=0, column=1, padx=5, pady=15)
        if self.student_list:
            self.student_combo.set("본인의 이름을 선택하세요")

        # 2️⃣ 가운데 블록 (폴더 열기 버튼)
        middle_section = tk.Frame(top_frame)
        middle_section.pack(side="left", padx=50, anchor="c")
        tk.Button(middle_section, text="촬영사진 폴더 열기 & 자동 분류\n(선택한 폴더 내 모든 사진 처리)", 
                  command=self.load_folder, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=30, height=2).pack()
        
        # 3️⃣ 오른쪽 블록 (4번째 칸 일괄입력 & 저장 버튼)
        right_section = tk.Frame(top_frame)
        right_section.pack(side="right", padx=10, anchor="n")

        self.bulk_suffix_frame = tk.Frame(right_section)
        self.bulk_suffix_frame.pack(side="top", pady=(0, 5))
        tk.Label(self.bulk_suffix_frame, text="4번째 칸 일괄입력:").pack(side="left")
        self.bulk_suffix_var = tk.StringVar()
        tk.Entry(self.bulk_suffix_frame, textvariable=self.bulk_suffix_var, width=8).pack(side="left", padx=2)
        tk.Button(self.bulk_suffix_frame, text="적용", command=self.apply_bulk_suffix, bg="#5bc0de", fg="white").pack(side="left")

        tk.Button(right_section, text="결과 저장", command=self.save_all, bg="#008CBA", fg="white", font=("Arial", 10, "bold"), width=25, height=1).pack(side="top", pady=2)

        # 메인 캔버스
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray90")
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="gray90")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.frame_id, width=e.width))
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    def setup_candidate_ui(self):
        cand_frame = tk.LabelFrame(self.root, text="실습 치아 선택 (AI 추론 후보 범위)", padx=10, pady=5)
        cand_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        self.cand_vars = {}
        nat_frame = tk.Frame(cand_frame)
        nat_frame.pack(side="left", padx=10, anchor="n")
        tk.Label(nat_frame, text="[자연치 대분류]").pack(anchor="w")
        nat_chk_frame = tk.Frame(nat_frame)
        nat_chk_frame.pack(anchor="w")
        nat_items = ["상악절치", "상악견치", "상악소구치", "상악제1대구치", "상악제2대구치", "하악절치", "하악견치", "하악소구치", "하악제1대구치", "하악제2대구치"]
        for idx, item in enumerate(nat_items):
            var = tk.BooleanVar()
            self.cand_vars[f"자연치_{item}"] = var
            tk.Checkbutton(nat_chk_frame, text=item, variable=var).grid(row=idx//5, column=idx%5, sticky="w")

        srt_frame = tk.Frame(cand_frame)
        srt_frame.pack(side="left", padx=20, anchor="n")
        tk.Label(srt_frame, text="[SRT 인공치]").pack(anchor="w")
        srt_chk_frame = tk.Frame(srt_frame)
        srt_chk_frame.pack(anchor="w")
        srt_items = [f"{i}{j}" for i in range(1, 5) for j in range(1, 9)]
        for idx, item in enumerate(srt_items):
            var = tk.BooleanVar()
            self.cand_vars[f"SRT_{item}"] = var
            tk.Checkbutton(srt_chk_frame, text=item, variable=var).grid(row=idx//8, column=idx%8, sticky="w")

    def apply_bulk_suffix(self):
        if not self.image_data:
            messagebox.showwarning("경고", "먼저 사진을 불러와 주세요.")
            return
        new_suffix = self.bulk_suffix_var.get().strip()
        for meta in self.image_data:
            meta["t_suffix"].set(new_suffix)

    def get_selected_candidates(self):
        return [name for name, var in self.cand_vars.items() if var.get()]

    def load_folder(self):
        # 💡 [추가] 콤보박스 선택 항목 파싱하여 교번과 이름 추출
        selection = self.student_combo.get()
        if not selection or selection == "본인의 이름을 선택하세요":
            messagebox.showwarning("경고", "먼저 엑셀 명단에서 본인의 이름을 선택해 주세요!")
            return
            
        student_id, student_name = self.student_mapping.get(selection, ("", ""))
        self.id_var.set(student_id)
        self.name_var.set(student_name)

        self.folder_path = filedialog.askdirectory(title="사진 폴더 선택")
        if not self.folder_path: return

        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.image_data.clear()

        files = [f for f in sorted(os.listdir(self.folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            messagebox.showinfo("알림", "선택한 폴더에 이미지 파일이 없습니다.")
            return

        total_photos = len(files)
        candidates = self.get_selected_candidates()
        
        progress_win = tk.Toplevel(self.root)
        progress_win.title("분석 중")
        progress_win.geometry("350x120")
        progress_win.transient(self.root) 
        progress_win.grab_set() 
        
        progress_label = tk.Label(progress_win, text="AI가 사진을 분석하고 있습니다...\n잠시만 기다려주세요.", font=("Arial", 10))
        progress_label.pack(pady=15)
        progress_bar = ttk.Progressbar(progress_win, orient="horizontal", length=280, mode="determinate")
        progress_bar.pack(pady=5)
        progress_bar["maximum"] = total_photos
        self.root.update() 

        row_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove", pady=5)
        row_frame.pack(fill="x", pady=5, padx=5, anchor="w")

        info_frame = tk.Frame(row_frame, width=150)
        info_frame.pack(side="left", fill="y", padx=10)
        tk.Label(info_frame, text=f"교번: {student_id}", font=("Arial", 12, "bold")).pack(anchor="w", pady=2)
        tk.Label(info_frame, text=f"이름: {student_name}", font=("Arial", 12, "bold")).pack(anchor="w", pady=2)
        tk.Label(info_frame, text=f"({total_photos}장 처리됨)", fg="blue").pack(anchor="w", pady=5)

        canvas_frame = tk.Frame(row_frame)
        canvas_frame.pack(side="left", fill="x", expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
        h_scroll.pack(side="bottom", fill="x")

        img_canvas = tk.Canvas(canvas_frame, height=280, bg="gray90")
        img_canvas.pack(side="top", fill="x", expand=True)

        img_canvas.configure(xscrollcommand=h_scroll.set)
        h_scroll.configure(command=img_canvas.xview)

        images_frame = tk.Frame(img_canvas, bg="gray90")
        img_canvas.create_window((0, 0), window=images_frame, anchor="nw")
        
        images_frame.bind("<Configure>", lambda e, c=img_canvas: c.configure(scrollregion=c.bbox("all")))

        current_progress = 0
        for file in files:
            img_path = os.path.join(self.folder_path, file)
            self.create_image_panel(images_frame, img_path, self.id_var, self.name_var, candidates)
            
            current_progress += 1
            progress_bar["value"] = current_progress
            progress_label.config(text=f"AI가 분석 중... ({current_progress}/{total_photos}장 완료)")
            self.root.update()

        progress_win.destroy()

    def create_image_panel(self, parent, img_path, id_var, name_var, candidates):
        panel = tk.Frame(parent, padx=10, pady=5, bg="gray90")
        panel.pack(side="left", anchor="n")

        p_type, p_num, p_dir, rot_label = self.ai.predict(img_path, candidates)
        abb_type = self.naming_rules.get(p_type, p_type)
        abb_num = self.naming_rules.get(p_num, p_num)
        abb_dir = self.naming_rules.get(p_dir, p_dir)
        auto_rotation_angle = (90 * rot_label) % 360 

        img_meta = {
            "path": img_path, "id_var": id_var, "name_var": name_var,
            "t_type": tk.StringVar(value=abb_type), 
            "t_num": tk.StringVar(value=abb_num),
            "v_dir": tk.StringVar(value=abb_dir), 
            "t_suffix": tk.StringVar(value=""),
            "rotation": auto_rotation_angle, 
            "flipped": False,
            "img_label": tk.Label(panel, cursor="hand2") 
        }
        self.image_data.append(img_meta)
        
        img_meta["img_label"].bind("<Button-1>", lambda e, m=img_meta: self.show_original_image(m))
        img_meta["img_label"].pack()
        
        self.update_thumbnail(img_meta)

        ctrl_frame = tk.Frame(panel, bg="gray90")
        ctrl_frame.pack(fill="x", pady=5)
        
        tk.Entry(ctrl_frame, textvariable=img_meta["t_type"], width=6, font=("Arial", 11)).pack(side="left", padx=1)
        tk.Entry(ctrl_frame, textvariable=img_meta["t_num"], width=7, font=("Arial", 11)).pack(side="left", padx=1) 
        tk.Entry(ctrl_frame, textvariable=img_meta["v_dir"], width=5, font=("Arial", 11)).pack(side="left", padx=1)
        tk.Entry(ctrl_frame, textvariable=img_meta["t_suffix"], width=5, font=("Arial", 11)).pack(side="left", padx=1) 

        btn_frame = tk.Frame(panel, bg="gray90")
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="90°", command=lambda m=img_meta: self.add_rotation(m, 90)).pack(side="left", expand=True, padx=1)
        tk.Button(btn_frame, text="180°", command=lambda m=img_meta: self.add_rotation(m, 180)).pack(side="left", expand=True, padx=1)
        tk.Button(btn_frame, text="반전", command=lambda m=img_meta: self.flip_image(m)).pack(side="left", expand=True, padx=1)

    def show_original_image(self, img_meta):
        top = tk.Toplevel(self.root)
        top.title("원본 사진 크게 보기")
        
        img = Image.open(img_meta["path"])
        img = ImageOps.exif_transpose(img)
        if img_meta["rotation"] != 0:
            img = img.rotate(-img_meta["rotation"], expand=True)
        if img_meta["flipped"]: 
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        screen_w = self.root.winfo_screenwidth() - 100
        screen_h = self.root.winfo_screenheight() - 100
        img.thumbnail((screen_w, screen_h))
        
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo 
        lbl.pack(padx=10, pady=10)

    def add_rotation(self, img_meta, angle):
        img_meta["rotation"] = (img_meta["rotation"] + angle) % 360
        self.update_thumbnail(img_meta)

    def flip_image(self, img_meta):
        img_meta["flipped"] = not img_meta["flipped"]
        self.update_thumbnail(img_meta)

    def update_thumbnail(self, img_meta):
        img = Image.open(img_meta["path"])
        img = ImageOps.exif_transpose(img)
        
        if img_meta["rotation"] != 0:
            img = img.rotate(-img_meta["rotation"], expand=True)
            
        if img_meta["flipped"]: 
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img.thumbnail((180, 180))
        photo = ImageTk.PhotoImage(img)
        img_meta["img_label"].config(image=photo)
        img_meta["img_label"].image = photo 

    def save_all(self):
        if not self.image_data: return
            
        result_dir = os.path.join(self.folder_path, "결과물")
        os.makedirs(result_dir, exist_ok=True)
        
        base_dir = get_executable_dir()
        dataset_archive_dir = os.path.join(base_dir, "Collected_Dataset")

        count = 0
        for meta in self.image_data:
            input_type = meta['t_type'].get().strip()
            input_num = meta['t_num'].get().strip()
            input_dir = meta['v_dir'].get().strip()
            input_suffix = meta['t_suffix'].get().strip()
            
            student_id = meta['id_var'].get().strip()
            student_name = meta['name_var'].get().strip()
            
            student_dir = os.path.join(result_dir, student_name)
            os.makedirs(student_dir, exist_ok=True)
            
            out_type = self.naming_rules.get(input_type, input_type)
            out_num = self.naming_rules.get(input_num, input_num)
            out_dir = self.naming_rules.get(input_dir, input_dir)
            
            base_name = f"{student_id}_{student_name}_{out_type}{out_num} {out_dir}"
            if input_suffix:
                base_name += f" {input_suffix}"
                
            ext = os.path.splitext(meta["path"])[1]
            new_fp = os.path.join(student_dir, f"{base_name}{ext}")
            
            dup = 1
            while os.path.exists(new_fp):
                new_fp = os.path.join(student_dir, f"{base_name}({dup}){ext}")
                dup += 1

            img = Image.open(meta["path"])
            img = ImageOps.exif_transpose(img)
            
            if meta["rotation"] != 0:
                img = img.rotate(-meta["rotation"], expand=True)
            if meta["flipped"]: 
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            img.save(new_fp, quality=100)
            count += 1
            
            full_type = self.reverse_rules.get(out_type, out_type)
            full_num = self.reverse_rules.get(out_num, out_num)
            full_dir = self.reverse_rules.get(out_dir, out_dir)
            
            label_folder_name = f"{full_type}_{full_num}_{full_dir}"
            specific_dataset_dir = os.path.join(dataset_archive_dir, label_folder_name)
            os.makedirs(specific_dataset_dir, exist_ok=True)
            
            archive_filename = f"{int(time.time() * 1000)}_{count}{ext}"
            archive_fp = os.path.join(specific_dataset_dir, archive_filename)
            img.save(archive_fp, quality=100)
            
        messagebox.showinfo("완료", f"원본 사진 폴더 안의 '결과물/{student_name}' 폴더에 최종 {count}장을 저장했습니다!\n\n(완전 자동 정렬 V2.1 개인용 적용 완료!)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAutoSorterApp_Individual(root)
    root.mainloop()
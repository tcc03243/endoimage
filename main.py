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


class DentalAutoSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("치과 방사선 사진 자동 정렬기 V2.1.2 (레이아웃 최적화)")
        self.root.geometry("1400x900")
        
        self.ai = DentalAIModel()
        self.image_data = []
        self.folder_path = ""
        self.student_data = {} 
        self.count_vars = [] 
        self.student_cards = [] 
        self.naming_rules = {} 
        self.reverse_rules = {} 
        
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
            
            if group not in self.student_data:
                self.student_data[group] = []
            self.student_data[group].append({"id": student_id, "name": name})

    def setup_ui(self):
        self.setup_candidate_ui()

        # 💡 [레이아웃 2줄로 분리]
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill="x", side="top")

        # --- 첫 번째 줄 (명단 및 순서 변경) ---
        row1_frame = tk.Frame(top_frame)
        row1_frame.pack(fill="x", side="top", anchor="nw")

        tk.Label(row1_frame, text="조 선택:").pack(side="left")
        self.group_combo = ttk.Combobox(row1_frame, values=list(self.student_data.keys()), state="readonly", width=10)
        self.group_combo.pack(side="left", padx=5)
        self.group_combo.bind("<<ComboboxSelected>>", self.on_group_select)

        tk.Label(row1_frame, text="조원별 사진 수:\n(드래그로 순서 변경)", justify="right").pack(side="left", padx=(20, 5))
        self.counts_frame = tk.Frame(row1_frame) 
        self.counts_frame.pack(side="left", fill="x", expand=True)

        # --- 두 번째 줄 (컨트롤 버튼들) ---
        row2_frame = tk.Frame(top_frame)
        row2_frame.pack(fill="x", side="top", pady=(15, 0))

        # (왼쪽) 일괄변경 & 폴더열기
        self.bulk_frame = tk.Frame(row2_frame)
        self.bulk_frame.pack(side="left")
        tk.Label(self.bulk_frame, text="사진수 일괄변경:").pack(side="left")
        self.bulk_var = tk.StringVar(value="8") 
        tk.Entry(self.bulk_frame, textvariable=self.bulk_var, width=4, justify="center").pack(side="left", padx=2)
        tk.Button(self.bulk_frame, text="적용", command=self.apply_bulk_count, bg="#f0ad4e", fg="white").pack(side="left")
        
        tk.Button(row2_frame, text="촬영사진 폴더 열기 & 자동 분류", command=self.load_folder, bg="#4CAF50", fg="white", width=25).pack(side="left", padx=30)
        
        # (오른쪽) 결과저장 & 4번째칸 일괄입력
        tk.Button(row2_frame, text="결과 저장", command=self.save_all, bg="#008CBA", fg="white", width=25).pack(side="right")

        self.bulk_suffix_frame = tk.Frame(row2_frame)
        self.bulk_suffix_frame.pack(side="right", padx=20)
        tk.Label(self.bulk_suffix_frame, text="4번째 칸 일괄입력:").pack(side="left")
        self.bulk_suffix_var = tk.StringVar()
        tk.Entry(self.bulk_suffix_frame, textvariable=self.bulk_suffix_var, width=8).pack(side="left", padx=2)
        tk.Button(self.bulk_suffix_frame, text="적용", command=self.apply_bulk_suffix, bg="#5bc0de", fg="white").pack(side="left")

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

    def on_group_select(self, event):
        group = self.group_combo.get()
        students = self.student_data.get(group, [])
        for widget in self.counts_frame.winfo_children():
            widget.destroy()
            
        self.count_vars.clear()
        self.student_cards.clear()
        
        for idx, student in enumerate(students):
            card_frame = tk.Frame(self.counts_frame, bd=1, relief="ridge", bg="white", cursor="fleur", padx=5, pady=2)
            card_frame.pack(side="left", padx=3)

            id_lbl = tk.Label(card_frame, text=student['id'], font=("Arial", 8, "bold"), fg="gray50", bg="white")
            id_lbl.pack()
            
            name_lbl = tk.Label(card_frame, text=student['name'], font=("Arial", 10, "bold"), bg="white")
            name_lbl.pack()

            var = tk.StringVar(value="6")
            self.count_vars.append(var)
            count_entry = tk.Entry(card_frame, textvariable=var, width=4, justify="center")
            count_entry.pack(pady=2)

            card_data = {
                "id": student['id'],
                "name": student['name'],
                "var": var,
                "frame": card_frame,
                "id_lbl": id_lbl,
                "name_lbl": name_lbl
            }
            self.student_cards.append(card_data)

            for w in (card_frame, id_lbl, name_lbl):
                w.bind("<ButtonPress-1>", lambda e, d=card_data: self.on_drag_start(e, d))
                w.bind("<B1-Motion>", self.on_drag_motion)
                w.bind("<ButtonRelease-1>", self.on_drag_release)

    def on_drag_start(self, event, card_data):
        self.dragged_card = card_data
        self.dragged_card["frame"].config(bg="#e0f7fa")
        self.dragged_card["id_lbl"].config(bg="#e0f7fa")
        self.dragged_card["name_lbl"].config(bg="#e0f7fa")

    def on_drag_motion(self, event):
        if not getattr(self, 'dragged_card', None): return
        x = event.x_root
        current_index = self.student_cards.index(self.dragged_card)
        target_index = current_index

        for i, card in enumerate(self.student_cards):
            if i == current_index: continue
            cx = card["frame"].winfo_rootx()
            cw = card["frame"].winfo_width()
            if cx <= x <= cx + cw:
                target_index = i
                break

        if target_index != current_index:
            item = self.student_cards.pop(current_index)
            self.student_cards.insert(target_index, item)
            
            for card in self.student_cards:
                card["frame"].pack_forget()
            for card in self.student_cards:
                card["frame"].pack(side="left", padx=3)
                
            self.count_vars = [c["var"] for c in self.student_cards]

    def on_drag_release(self, event):
        if getattr(self, 'dragged_card', None):
            self.dragged_card["frame"].config(bg="white")
            self.dragged_card["id_lbl"].config(bg="white")
            self.dragged_card["name_lbl"].config(bg="white")
            self.dragged_card = None

    def apply_bulk_count(self):
        new_count = self.bulk_var.get().strip()
        if not new_count.isdigit():
            messagebox.showwarning("경고", "숫자만 입력해 주세요.")
            return
        for var in self.count_vars:
            var.set(new_count)

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
        group = self.group_combo.get()
        if not group:
            messagebox.showwarning("경고", "먼저 조를 선택해주세요.")
            return
            
        students = [{"id": card["id"], "name": card["name"]} for card in self.student_cards]
        photo_counts = []
        for var in self.count_vars:
            val = var.get().strip()
            if val.isdigit(): photo_counts.append(int(val))
            else: return

        self.folder_path = filedialog.askdirectory(title="사진 폴더 선택")
        if not self.folder_path: return

        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.image_data.clear()

        files = [f for f in sorted(os.listdir(self.folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_photos = sum(photo_counts)
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

        file_idx = 0
        current_progress = 0
        
        for student, count in zip(students, photo_counts):
            user_files = files[file_idx:file_idx+count]
            file_idx += count
            
            row_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove", pady=5)
            row_frame.pack(fill="x", pady=5, padx=5, anchor="w")

            info_frame = tk.Frame(row_frame, width=150)
            info_frame.pack(side="left", fill="y", padx=10)
            tk.Label(info_frame, text=f"교번: {student['id']}", font=("Arial", 10, "bold")).pack(anchor="w")
            tk.Label(info_frame, text=f"이름: {student['name']}", font=("Arial", 10, "bold")).pack(anchor="w")
            tk.Label(info_frame, text=f"({count}장)").pack(anchor="w")

            id_var = tk.StringVar(value=student['id'])
            name_var = tk.StringVar(value=student['name'])

            canvas_frame = tk.Frame(row_frame)
            canvas_frame.pack(side="left", fill="x", expand=True)

            h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
            h_scroll.pack(side="bottom", fill="x")

            # 💡 [조정] 사진 크기가 줄었으니 캔버스 높이도 280 -> 240으로 다이어트!
            img_canvas = tk.Canvas(canvas_frame, height=240, bg="gray90")
            img_canvas.pack(side="top", fill="x", expand=True)

            img_canvas.configure(xscrollcommand=h_scroll.set)
            h_scroll.configure(command=img_canvas.xview)

            images_frame = tk.Frame(img_canvas, bg="gray90")
            img_canvas.create_window((0, 0), window=images_frame, anchor="nw")
            
            images_frame.bind("<Configure>", lambda e, c=img_canvas: c.configure(scrollregion=c.bbox("all")))

            for file in user_files:
                img_path = os.path.join(self.folder_path, file)
                self.create_image_panel(images_frame, img_path, id_var, name_var, candidates)
                
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
        
        tk.Entry(ctrl_frame, textvariable=img_meta["t_type"], width=5, font=("Arial", 11)).pack(side="left", padx=1)
        tk.Entry(ctrl_frame, textvariable=img_meta["t_num"], width=6, font=("Arial", 11)).pack(side="left", padx=1) 
        tk.Entry(ctrl_frame, textvariable=img_meta["v_dir"], width=4, font=("Arial", 11)).pack(side="left", padx=1)
        tk.Entry(ctrl_frame, textvariable=img_meta["t_suffix"], width=4, font=("Arial", 11)).pack(side="left", padx=1) 

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
            
        # 💡 [핵심] 이미지 크기를 180x180에서 144x144(80%)로 축소 
        img.thumbnail((144, 144))
        photo = ImageTk.PhotoImage(img)
        img_meta["img_label"].config(image=photo)
        img_meta["img_label"].image = photo 

    def save_all(self):
        if not self.image_data: return
        
        group_name = self.group_combo.get()
        if not group_name: group_name = "결과물"
            
        original_result_dir = os.path.join(self.folder_path, group_name)
        result_dir = original_result_dir
        folder_dup = 1
        while os.path.exists(result_dir):
            result_dir = f"{original_result_dir}({folder_dup})"
            folder_dup += 1
            
        os.makedirs(result_dir, exist_ok=True)
        
        base_dir = get_executable_dir()
        dataset_archive_dir = os.path.join(base_dir, "Collected_Dataset")

        count = 0
        for meta in self.image_data:
            input_type = meta['t_type'].get().strip()
            input_num = meta['t_num'].get().strip()
            input_dir = meta['v_dir'].get().strip()
            input_suffix = meta['t_suffix'].get().strip()
            student_name = meta['name_var'].get().strip()
            
            student_dir = os.path.join(result_dir, student_name)
            os.makedirs(student_dir, exist_ok=True)
            
            out_type = self.naming_rules.get(input_type, input_type)
            out_num = self.naming_rules.get(input_num, input_num)
            out_dir = self.naming_rules.get(input_dir, input_dir)
            
            base_name = f"{meta['id_var'].get()}_{student_name}_{out_type}{out_num} {out_dir}"
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
            
        saved_folder_name = os.path.basename(result_dir)
        messagebox.showinfo("완료", f"'{saved_folder_name}' 폴더에 조원들의 실습 결과물 {count}장을 저장했습니다!\n\n(완전 자동 정렬 V2.1.2 적용 완료!)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAutoSorterApp(root)
    root.mainloop()
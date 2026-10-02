import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import timm
import openpyxl
import time

def get_executable_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))

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
            print(f"[자동 생성] class_names.txt 파일이 생성되었습니다. (총 {len(self.class_names)}개)")

    def load_model(self):
        model_path = os.path.join(get_executable_dir(), "dental_model.pth")
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                actual_num_classes = state_dict['fc.weight'].shape[0]
                
                self.model = timm.create_model('resnet18', pretrained=False, num_classes=actual_num_classes) 
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ 모델 로딩 완벽 성공!")
            else:
                print("경고: dental_model.pth 가중치 파일이 없습니다.")
        except Exception as e:
            print(f"모델 로딩 에러: {e}")

    def predict(self, image_path, candidates):
        if self.model is None or not self.class_names:
            return "미상", "미상", "미상"
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor) 
                
                if candidates:
                    valid_indices = []
                    for i, name in enumerate(self.class_names):
                        if any(name.startswith(cand) for cand in candidates):
                            valid_indices.append(i)
                    
                    if valid_indices:
                        mask = torch.ones_like(outputs, dtype=torch.bool)
                        mask[0, valid_indices] = False
                        outputs[mask] = -float('inf')
                
                _, predicted = torch.max(outputs, 1)
                idx = predicted.item()
                
            predicted_label = self.class_names[idx]
            
            parts = predicted_label.split('_')
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
            return predicted_label, "", ""
            
        except Exception as e:
            print(f"추론 중 에러 발생: {e}")
            return "에러", "에러", "에러"

class DentalAutoSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("치과 방사선 사진 자동 정렬기 (조장용)")
        self.root.geometry("1400x800")
        
        self.ai = DentalAIModel()
        self.image_data = []
        self.folder_path = ""
        self.student_data = {} 
        self.count_vars = [] 
        self.naming_rules = {} 
        
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
                f.write("# 왼쪽에는 폴더명(원본) = 오른쪽에는 변환될 이름(약어)을 적어주세요.\n\n")
                for k, v in default_rules.items():
                    f.write(f"{k}={v}\n")
            self.naming_rules = default_rules
            print("[자동 생성] naming_rules.txt 파일이 생성되었습니다.")
        else:
            with open(rule_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): 
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        self.naming_rules[k.strip()] = v.strip()

    def load_excel_data(self):
        base_dir = get_executable_dir()
        excel_path = os.path.join(base_dir, "students.xlsx")
        
        if not os.path.exists(excel_path):
            messagebox.showwarning("경고", f"실행 파일과 같은 위치에 'students.xlsx'가 없습니다.\n임시 명단으로 빈칸이 표시됩니다.")
            return

        wb = openpyxl.load_workbook(excel_path, data_only=True)
        ws = wb.active
        
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row[0]: continue
            group = str(row[0])
            student_id = str(row[1]) if row[1] else ""
            name = str(row[2]) if row[2] else ""
            
            if group not in self.student_data:
                self.student_data[group] = []
            self.student_data[group].append({"id": student_id, "name": name})

    def setup_ui(self):
        self.setup_candidate_ui()

        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill="x", side="top")

        tk.Label(top_frame, text="조 선택:").pack(side="left")
        self.group_combo = ttk.Combobox(top_frame, values=list(self.student_data.keys()), state="readonly", width=10)
        self.group_combo.pack(side="left", padx=5)
        self.group_combo.bind("<<ComboboxSelected>>", self.on_group_select)

        tk.Label(top_frame, text="조원별 사진 수:").pack(side="left", padx=(20, 5))
        self.counts_frame = tk.Frame(top_frame) 
        self.counts_frame.pack(side="left")
        
        tk.Button(top_frame, text="촬영사진 폴더 열기 & 자동 분류", command=self.load_folder, bg="#4CAF50", fg="white").pack(side="left", padx=30)
        tk.Button(top_frame, text="결과 저장", command=self.save_all, bg="#008CBA", fg="white").pack(side="right", padx=10)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray90")
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="gray90")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
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
        for idx, student in enumerate(students):
            tk.Label(self.counts_frame, text=student['name'], font=("Arial", 9, "bold")).grid(row=0, column=idx, padx=5)
            var = tk.StringVar(value="6")
            self.count_vars.append(var)
            tk.Entry(self.counts_frame, textvariable=var, width=5, justify="center").grid(row=1, column=idx, padx=5, pady=2)

    def get_selected_candidates(self):
        return [name for name, var in self.cand_vars.items() if var.get()]

    def load_folder(self):
        group = self.group_combo.get()
        if not group:
            messagebox.showwarning("경고", "먼저 조를 선택해주세요.")
            return
            
        students = self.student_data.get(group, [])
        photo_counts = []
        for var in self.count_vars:
            val = var.get().strip()
            if val.isdigit():
                photo_counts.append(int(val))
            else:
                messagebox.showwarning("경고", "사진 수는 반드시 숫자로 입력해야 합니다.")
                return
                
        if len(photo_counts) != len(students):
            messagebox.showwarning("경고", f"입력된 데이터에 문제가 있습니다.")
            return

        self.folder_path = filedialog.askdirectory(title="사진 폴더 선택")
        if not self.folder_path: return

        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.image_data.clear()

        files = [f for f in sorted(os.listdir(self.folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_photos = sum(photo_counts)
        if total_photos > len(files):
            messagebox.showwarning("경고", f"설정한 사진 수의 합({total_photos}장)이 폴더의 실제 사진 수({len(files)}장)보다 많습니다.")
            return

        candidates = self.get_selected_candidates()
        
        # --- 로딩바 UI 생성 ---
        progress_win = tk.Toplevel(self.root)
        progress_win.title("분석 중")
        progress_win.geometry("350x120")
        progress_win.transient(self.root) # 메인 윈도우 위에 고정
        progress_win.grab_set() # 팝업이 떠 있는 동안 메인 윈도우 클릭 방지
        
        # 팝업을 화면 중앙에 배치
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - 175
        y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - 60
        progress_win.geometry(f"+{x}+{y}")
        
        progress_label = tk.Label(progress_win, text="AI가 사진을 분석하고 있습니다...\n잠시만 기다려주세요.", font=("Arial", 10))
        progress_label.pack(pady=15)
        
        progress_bar = ttk.Progressbar(progress_win, orient="horizontal", length=280, mode="determinate")
        progress_bar.pack(pady=5)
        progress_bar["maximum"] = total_photos
        
        self.root.update() # UI 강제 업데이트
        # ----------------------

        file_idx = 0
        current_progress = 0
        
        for student, count in zip(students, photo_counts):
            user_files = files[file_idx:file_idx+count]
            file_idx += count
            
            row_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove", pady=5)
            row_frame.pack(fill="x", pady=5, padx=5, anchor="w")

            info_frame = tk.Frame(row_frame, width=150)
            info_frame.pack(side="left", fill="y", padx=10)
            tk.Label(info_frame, text=f"교번: {student['id']}").pack(anchor="w")
            tk.Label(info_frame, text=f"이름: {student['name']}").pack(anchor="w")
            tk.Label(info_frame, text=f"({count}장)").pack(anchor="w")

            id_var = tk.StringVar(value=student['id'])
            name_var = tk.StringVar(value=student['name'])

            images_frame = tk.Frame(row_frame)
            images_frame.pack(side="left", fill="x", expand=True)

            for file in user_files:
                img_path = os.path.join(self.folder_path, file)
                self.create_image_panel(images_frame, img_path, id_var, name_var, candidates)
                
                # --- 로딩 진행도 업데이트 ---
                current_progress += 1
                progress_bar["value"] = current_progress
                progress_label.config(text=f"AI가 사진을 분석하고 있습니다... ({current_progress}/{total_photos}장 완료)")
                self.root.update() # 실시간으로 화면 갱신
                # --------------------------

        progress_win.destroy() # 로딩 완료 후 창 닫기

    def create_image_panel(self, parent, img_path, id_var, name_var, candidates):
        panel = tk.Frame(parent, padx=5, pady=5)
        panel.pack(side="left", anchor="n")

        tooth_type, tooth_num, view_dir = self.ai.predict(img_path, candidates)
        
        rotation_angle = 0 

        img_meta = {
            "path": img_path, "id_var": id_var, "name_var": name_var,
            "t_type": tk.StringVar(value=tooth_type), "t_num": tk.StringVar(value=tooth_num),
            "v_dir": tk.StringVar(value=view_dir), "rotation": rotation_angle, "flipped": False,
            "img_label": tk.Label(panel)
        }
        self.image_data.append(img_meta)
        
        img_meta["img_label"].pack()
        self.update_thumbnail(img_meta)

        ctrl_frame = tk.Frame(panel)
        ctrl_frame.pack(fill="x", pady=2)
        tk.Entry(ctrl_frame, textvariable=img_meta["t_type"], width=6).pack(side="left")
        tk.Entry(ctrl_frame, textvariable=img_meta["t_num"], width=9).pack(side="left") 
        tk.Entry(ctrl_frame, textvariable=img_meta["v_dir"], width=4).pack(side="left")

        btn_frame = tk.Frame(panel)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="90°", command=lambda m=img_meta: self.add_rotation(m, 90)).pack(side="left", expand=True)
        tk.Button(btn_frame, text="180°", command=lambda m=img_meta: self.add_rotation(m, 180)).pack(side="left", expand=True)
        tk.Button(btn_frame, text="↔", command=lambda m=img_meta: self.flip_image(m)).pack(side="left", expand=True)

    def add_rotation(self, img_meta, angle):
        img_meta["rotation"] = (img_meta["rotation"] + angle) % 360
        self.update_thumbnail(img_meta)

    def flip_image(self, img_meta):
        img_meta["flipped"] = not img_meta["flipped"]
        self.update_thumbnail(img_meta)

    def update_thumbnail(self, img_meta):
        img = Image.open(img_meta["path"])
        
        if img_meta["rotation"] != 0:
            img = img.rotate(-img_meta["rotation"], expand=True)
            
        if img_meta["flipped"]: 
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img.thumbnail((120, 120))
        photo = ImageTk.PhotoImage(img)
        img_meta["img_label"].config(image=photo)
        img_meta["img_label"].image = photo 

    def save_all(self):
        if not self.image_data: return
        
        group_name = self.group_combo.get()
        if not group_name:
            group_name = "결과물"
            
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
            final_type = meta['t_type'].get().strip()
            final_num = meta['t_num'].get().strip()
            final_dir = meta['v_dir'].get().strip()
            student_name = meta['name_var'].get().strip()
            
            student_dir = os.path.join(result_dir, student_name)
            os.makedirs(student_dir, exist_ok=True)
            
            out_type = self.naming_rules.get(final_type, final_type)
            out_num = self.naming_rules.get(final_num, final_num)
            out_dir = self.naming_rules.get(final_dir, final_dir)
            
            base_name = f"{meta['id_var'].get()}_{student_name}_{out_type}{out_num} {out_dir}"
            ext = os.path.splitext(meta["path"])[1]
            new_fp = os.path.join(student_dir, f"{base_name}{ext}")
            
            dup = 1
            while os.path.exists(new_fp):
                new_fp = os.path.join(student_dir, f"{base_name}({dup}){ext}")
                dup += 1

            img = Image.open(meta["path"])
            
            if meta["rotation"] != 0:
                img = img.rotate(-meta["rotation"], expand=True)
                
            if meta["flipped"]: 
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            img.save(new_fp, quality=100)
            count += 1
            
            label_folder_name = f"{final_type}_{final_num}_{final_dir}"
            specific_dataset_dir = os.path.join(dataset_archive_dir, label_folder_name)
            os.makedirs(specific_dataset_dir, exist_ok=True)
            
            archive_filename = f"{int(time.time() * 1000)}_{count}{ext}"
            archive_fp = os.path.join(specific_dataset_dir, archive_filename)
            img.save(archive_fp, quality=100)
            
        saved_folder_name = os.path.basename(result_dir)
        messagebox.showinfo("완료", f"'{saved_folder_name}' 폴더에 조원들의 실습 결과물 {count}장을 저장했습니다!\n\n(미래 AI 학습용 데이터도 자동 수집되었습니다.)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAutoSorterApp(root)
    root.mainloop()
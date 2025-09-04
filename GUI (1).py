import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

# ---------------------------
# 1. Kiến trúc mô hình
# ---------------------------
class BertCNNClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout=0.1):
        super(BertCNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3),
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=4),
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128*3, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state.transpose(1, 2)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ---------------------------
# 2. Load tokenizer + model
# ---------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertCNNClassifier()
model.load_state_dict(torch.load('models/model_fake_news/model_state2.pt', map_location=torch.device('cpu')))
model.eval()

# ---------------------------
# 3. Hàm dự đoán
# ---------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return ("Thật" if pred == 1 else "Giả"), confidence

# ---------------------------
# 4. Tkinter GUI
# ---------------------------
df = None
max_rows = 0

def load_csv():
    global df, max_rows
    file_path = filedialog.askopenfilename(title="Chọn file CSV", filetypes=[("CSV files","*.csv")])
    if not file_path:
        return
    try:
        df = pd.read_csv(file_path)
        max_rows = len(df)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể đọc file CSV:\n{e}")
        return
    info_label.config(text=f"Đã tải CSV: {max_rows} dòng, {len(df.columns)} cột.")

def predict_rows():
    if df is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng tải CSV trước!")
        return

    indices_text = row_entry.get().strip()
    if not indices_text:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập số dòng hoặc khoảng dòng (vd: 1,3,5 hoặc 2-4)!")
        return

    indices = []
    try:
        parts = indices_text.split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                indices.extend(range(int(start)-1, int(end)))
            else:
                indices.append(int(part)-1)
    except ValueError:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập định dạng số hợp lệ!")
        return

    indices = [i for i in indices if 0 <= i < max_rows]
    if not indices:
        messagebox.showwarning("Cảnh báo", f"Số dòng phải từ 1 đến {max_rows}!")
        return

    # Tạo dataframe chỉ chứa các dòng cần dự đoán
    df_selected = df.iloc[indices].copy()
    df_selected['Kết quả'] = ""
    df_selected['Xác suất (%)'] = ""

    for i, idx in enumerate(indices):
        if len(df.columns) < 4:
            messagebox.showwarning("Cảnh báo", "CSV phải có ít nhất 4 cột để dự đoán!")
            return
        text = " ".join([str(df.iloc[idx, j]) for j in range(4)])
        res, conf = predict(text)
        df_selected.iloc[i, df_selected.columns.get_loc('Kết quả')] = res
        df_selected.iloc[i, df_selected.columns.get_loc('Xác suất (%)')] = round(conf*100, 2)

    save_path = filedialog.asksaveasfilename(
        title="Lưu file kết quả",
        defaultextension=".csv",
        filetypes=[("CSV files","*.csv")],
        initialfile="result.csv"
    )
    if save_path:
        try:
            df_selected.to_csv(save_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Hoàn tất", f"Đã lưu kết quả {len(indices)} dòng ra:\n{save_path}")
        except PermissionError:
            messagebox.showerror("Lỗi", "Không thể ghi file. Vui lòng đóng file nếu đang mở hoặc chọn thư mục khác.")

# ---------------------------
# 5. GUI đẹp hơn
# ---------------------------
root = tk.Tk()
root.title("Nhận diện tin tức Thật/Giả")
root.geometry("550x300")
root.configure(bg="#eef2f3")

# Header
header = tk.Label(root, text="📰 Nhận diện tin tức Thật/Giả", font=("Segoe UI", 18, "bold"), bg="#eef2f3", fg="#2c3e50")
header.pack(pady=15)

# Chọn CSV
csv_frame = tk.Frame(root, bg="#eef2f3")
csv_frame.pack(pady=5)
tk.Label(csv_frame, text="Chọn file CSV:", font=("Segoe UI", 12), bg="#eef2f3").pack(side="left", padx=5)
tk.Button(csv_frame, text="Tải CSV", font=("Segoe UI", 12, "bold"), bg="#27ae60", fg="white", padx=10, pady=5, bd=0, command=load_csv).pack(side="left", padx=5)
info_label = tk.Label(root, text="Chưa tải CSV", font=("Segoe UI", 11), bg="#eef2f3", fg="#34495e")
info_label.pack(pady=5)

# Nhập dòng
row_frame = tk.Frame(root, bg="#eef2f3")
row_frame.pack(pady=10)
tk.Label(row_frame, text="Dòng cần dự đoán (vd: 1,3,5 hoặc 2-4):", font=("Segoe UI", 12), bg="#eef2f3").pack(side="left", padx=5)
row_entry = tk.Entry(row_frame, font=("Segoe UI", 12), width=20, bd=2, relief="groove")
row_entry.pack(side="left", padx=5)

# Nút dự đoán
tk.Button(root, text="Dự đoán & Lưu kết quả", font=("Segoe UI", 12, "bold"), bg="#2980b9", fg="white", padx=10, pady=7, bd=0, command=predict_rows).pack(pady=20)

root.mainloop()

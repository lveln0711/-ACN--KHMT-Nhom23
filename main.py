import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.modeling_outputs import SequenceClassifierOutput

# 1. Kiểm tra PyTorch & CUDA
print(f" PyTorch version: {torch.__version__}")
print(f" CUDA runtime version: {torch.version.cuda}")
print(f" GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
   print(f" GPU name: {torch.cuda.get_device_name(0)}")
else:
   print(" Đang chạy trên CPU - sẽ chậm hơn so với GPU")

# 2. Cấu hình đường dẫn file
MODEL_PATH = "models/model_fake_news"   # Thư mục lưu mô hình và tokenizer
TOKENIZER_PATH = "tokenizer/"
TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/val.csv"
TEST_FILE = "data/processed/test.csv"
LABELS_TEST_FILE = "data/processed/labels_test.csv"  # Thêm file nhãn thật test
RESULTS_DIR = "results"
PREDICT_FILE = os.path.join(RESULTS_DIR, "predicts.csv")
CONF_MATRIX_FILE = os.path.join(RESULTS_DIR, "confusion_matrix.png")

MODEL_NAME = "bert-base-uncased"

# 3. Định nghĩa Dataset
class NewsDataset(Dataset):
   def __init__(self, dataframe, tokenizer, max_len=256, has_labels=True):
       self.data = dataframe
       self.tokenizer = tokenizer
       self.max_len = max_len
       self.has_labels = has_labels

   def __len__(self):
       return len(self.data)

   def __getitem__(self, index):
       title = str(self.data.iloc[index].get('title', ''))
       text = str(self.data.iloc[index].get('text', ''))

       inputs = self.tokenizer(
           title + " " + text,
           padding="max_length",
           truncation=True,
           max_length=self.max_len,
           return_tensors="pt"
       )

       item = {
           "input_ids": inputs["input_ids"].squeeze(),
           "attention_mask": inputs["attention_mask"].squeeze(),
       }

       if self.has_labels and 'label' in self.data.columns:
           label = int(self.data.iloc[index]['label'])
           item["labels"] = torch.tensor(label, dtype=torch.long)

       return item

# 4. Định nghĩa mô hình BERT + CNN
class BertCNNClassifier(nn.Module):
   def __init__(self, pretrained_model_name, num_labels=2):
       super(BertCNNClassifier, self).__init__()
       self.bert = AutoModel.from_pretrained(pretrained_model_name)
       self.dropout = nn.Dropout(0.3)

       self.convs = nn.ModuleList([
           nn.Conv1d(in_channels=768, out_channels=128, kernel_size=k)
           for k in [3,4,5]
       ])
       self.fc = nn.Linear(128 * 3, num_labels)

       self.loss_fct = nn.CrossEntropyLoss()

   def forward(self, input_ids, attention_mask, labels=None):
       outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
       x = outputs[0].permute(0, 2, 1)  # (batch, channels=768, seq_len)

       conv_outs = [F.relu(conv(x)) for conv in self.convs]
       pooled = [F.max_pool1d(c_out, kernel_size=c_out.shape[2]).squeeze(2) for c_out in conv_outs]
       cat = torch.cat(pooled, dim=1)
       drop = self.dropout(cat)
       logits = self.fc(drop)

       loss = None
       if labels is not None:
           loss = self.loss_fct(logits, labels)

       return SequenceClassifierOutput(
           loss=loss,
           logits=logits,
       )

# 5. Hàm tính metrics
def compute_metrics(pred):
   labels = pred.label_ids
   preds = pred.predictions.argmax(-1)
   acc = accuracy_score(labels, preds)
   precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
   roc_auc = roc_auc_score(labels, preds)
   return {
       'accuracy': acc,
       'precision': precision,
       'recall': recall,
       'f1': f1,
       'roc_auc': roc_auc
   }

# 6. Load Tokenizer
if os.path.exists(TOKENIZER_PATH):
   tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
   print(" Đã load tokenizer từ thư mục")
else:
   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
   tokenizer.save_pretrained(TOKENIZER_PATH)
   print(" Đã tải và lưu tokenizer mới")

# 7. Hàm huấn luyện và lưu mô hình
def train_and_save():
   train_df = pd.read_csv(TRAIN_FILE)
   val_df = pd.read_csv(VAL_FILE)

   train_dataset = NewsDataset(train_df, tokenizer)
   val_dataset = NewsDataset(val_df, tokenizer)

   model = BertCNNClassifier(MODEL_NAME, num_labels=2)

   training_args = TrainingArguments(
       output_dir="./results",
       learning_rate=2e-5,
       per_device_train_batch_size=8,
       per_device_eval_batch_size=8,
       num_train_epochs=3,
       weight_decay=0.01,
       logging_dir="./logs",
       logging_steps=50,
   )
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       tokenizer=tokenizer,
       compute_metrics=compute_metrics,
   )
   trainer.train()

   # Tạo thư mục lưu mô hình nếu chưa có
   os.makedirs(MODEL_PATH, exist_ok=True)
   # Lưu state_dict mô hình (cả BERT + CNN)
   torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model_state.pt"))
   # Lưu tokenizer vào cùng thư mục mô hình
   tokenizer.save_pretrained(MODEL_PATH)

   print(f" Đã lưu mô hình vào thư mục {MODEL_PATH}")

   return model

# 8. Hàm load mô hình đã lưu
def load_model():
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = BertCNNClassifier(MODEL_NAME, num_labels=2)
   model_path_file = os.path.join(MODEL_PATH, "model_state.pt")

   if not os.path.exists(model_path_file):
       raise FileNotFoundError(f"Không tìm thấy file model_state.pt tại {model_path_file}")

   model.load_state_dict(torch.load(model_path_file, map_location=device))
   model.to(device)

   global tokenizer
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

   print(" Load mô hình và tokenizer thành công")
   return model

# 9. Hàm vẽ confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, save_path):
   cm = confusion_matrix(true_labels, pred_labels)
   plt.figure(figsize=(6,6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
   plt.xlabel('Dự đoán')
   plt.ylabel('Thực tế')
   plt.title('Confusion Matrix')
   plt.savefig(save_path)
   plt.close()
   print(f"Đã lưu ảnh confusion matrix vào {save_path}")

# 10. Hàm đánh giá mô hình trên tập val
def evaluate_model(model):
   print(" Bắt đầu đánh giá mô hình trên tập validation...")
   val_df = pd.read_csv(VAL_FILE)
   val_dataset = NewsDataset(val_df, tokenizer)
   trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

   results = trainer.predict(val_dataset)
   preds = torch.argmax(torch.tensor(results.predictions), dim=1).tolist()
   labels = val_df['label'].tolist()

   acc = accuracy_score(labels, preds)
   precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
   roc_auc = roc_auc_score(labels, preds)
   print("======= Đánh giá mô hình =======")
   print(f"Accuracy : {acc:.4f}")
   print(f"Precision: {precision:.4f}")
   print(f"Recall   : {recall:.4f}")
   print(f"F1-score : {f1:.4f}")
   print(f"ROC AUC  : {roc_auc:.4f}")
   print("Confusion Matrix:")
   print(confusion_matrix(labels, preds))

   plot_confusion_matrix(labels, preds, CONF_MATRIX_FILE)
   print("===============================")

# 11. Hàm đánh giá trên file predicts.csv
def evaluate_predictions_file(predict_file):
    if not os.path.exists(predict_file):
        print(f"File dự đoán {predict_file} không tồn tại, không thể đánh giá.")
        return

    df = pd.read_csv(predict_file)
    if 'label' not in df.columns:
        print(f"File {predict_file} không có cột 'label' (nhãn thật), không thể đánh giá.")
        return
    if 'label_pred' not in df.columns:
        print(f"File {predict_file} không có cột 'label_pred' (nhãn dự đoán), không thể đánh giá.")
        return

    labels = df['label'].tolist()
    preds = df['label_pred'].tolist()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    roc_auc = roc_auc_score(labels, preds)

    print("======= Đánh giá mô hình trên kết quả dự đoán trong predicts.csv =======")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_predicts.png")
    plot_confusion_matrix(labels, preds, cm_path)
    print("===============================")

# 12. Hàm main
def main():
   if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "model_state.pt")):
       print(" Chưa có mô hình, bắt đầu huấn luyện...")
       model = train_and_save()
   else:
       model = load_model()

   # Kiểm tra file test và file nhãn thật
   if not os.path.exists(TEST_FILE):
       raise FileNotFoundError(f"Không tìm thấy file test.csv tại {TEST_FILE}")
   if not os.path.exists(LABELS_TEST_FILE):
       raise FileNotFoundError(f"Không tìm thấy file labels_test.csv tại {LABELS_TEST_FILE}")

   # Đọc dữ liệu test và nhãn thật test
   test_df = pd.read_csv(TEST_FILE)
   true_labels_df = pd.read_csv(LABELS_TEST_FILE)

   # Kiểm tra số dòng khớp
   if len(test_df) != len(true_labels_df):
       raise ValueError("Số dòng trong test.csv và labels_test.csv không khớp!")

   # Gộp nhãn thật vào test_df
   test_df['label'] = true_labels_df['label']

   # Tạo dataset để dự đoán (has_labels=False để Trainer không dùng nhãn thật trong prediction)
   test_dataset = NewsDataset(test_df, tokenizer, has_labels=False)

   trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
   predictions = trainer.predict(test_dataset)
   preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).tolist()

   # Thêm nhãn dự đoán
   test_df['label_pred'] = preds

   # Lưu kết quả gộp vào predicts.csv
   os.makedirs(RESULTS_DIR, exist_ok=True)
   test_df.to_csv(PREDICT_FILE, index=False, encoding='utf-8')
   print(f" Đã lưu kết quả dự đoán và nhãn thật vào {PREDICT_FILE}")

   # Đánh giá dựa trên file predicts.csv
   evaluate_predictions_file(PREDICT_FILE)

if __name__ == "__main__":
   main()

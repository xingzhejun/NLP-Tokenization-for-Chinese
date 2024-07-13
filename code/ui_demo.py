import tkinter as tk
from tkinter import simpledialog, messagebox
from fmm import FMMSegment
from data_process import HMMDataLoader
from hmm import HMM
from unigram_model import UniGramSeg



vocabulary_path = 'icwb2-data/gold/pku_training_words.utf8'
data_path = 'icwb2-data/training/pku_training.utf8'
fmm = FMMSegment(vocabulary_path)

hmm = HMM(data_path)
A, B, Pi = hmm.build_supervised_model()

unigram = UniGramSeg()
data_loader = HMMDataLoader(data_path)
vocab_dict = data_loader.generate_vocab_dict()
unigram.set_dict(vocab_dict)


# Function to handle algorithm selection
def select_algorithm():
    algorithm = simpledialog.askstring("请选择算法，输入对应数字", "请选择 (1代表fmm, 2代表unigram, 3代表hmm): ")
    if algorithm == "1":
        result = fmm.cut(user_input.get())
        output_result.set("/".join(result))
    elif algorithm == "2":
        result = unigram.cut(user_input.get())
        output_result.set("/".join(result))
    elif algorithm == "3":
        result = hmm.cut(user_input.get())
        output_result.set("/".join(result))
    else:
        messagebox.showerror("Error", "Invalid algorithm choice")


# Create the main window
root = tk.Tk()
root.title("中文分词")

# Input text box
user_input = tk.StringVar()
input_label = tk.Label(root, text="请输入一段话:", font=("Arial",10))
input_label.config(height=2, width=30)
input_label.pack()
input_entry = tk.Entry(root, textvariable=user_input)
input_entry.config(width=30)
input_entry.pack()

# Algorithm selection button
algorithm_button = tk.Button(root, text="请选择一个算法", command=select_algorithm)
algorithm_button.config(height=1, width=30)
algorithm_button.pack()

output_result = tk.StringVar()
output_label = tk.Label(root, text="中文分词结果:",font=("Arial",10))
output_label.config(height=2,width=30)
output_label.pack()
output_entry = tk.Entry(root, textvariable=output_result, state='disabled')
input_entry.config(width=30)
output_entry.pack()

# Run the UI
root.geometry("400x200")
root.mainloop()
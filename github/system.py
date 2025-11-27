import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
def load_expression_matrix_from_wide_table(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    if 'DrugName' in df.columns:
        df.set_index('DrugName', inplace=True)
    else:
        messagebox.showerror("错误", "没有找到名为 'DrugName' 的列，检查数据格式。")
        return None
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')
    return df
def load_gene_set_from_symbol_files(up_path, down_path):
    up_genes = pd.read_csv(up_path)["gene_id"].astype(str).str.upper().dropna().tolist()
    down_genes = pd.read_csv(down_path)["gene_id"].astype(str).str.upper().dropna().tolist()
    return up_genes, down_genes
def compute_gsea_score(ranked_genes, gene_set, weighted_score_type=1):
    N = len(ranked_genes)
    hits = np.in1d(ranked_genes, list(gene_set), assume_unique=True)
    Nh = hits.sum()
    if Nh == 0:
        return 0.0
    no_hits = ~hits
    if weighted_score_type == 0:
        Phit = np.cumsum(hits) / Nh
    else:
        weights = np.abs(np.linspace(1.0, 0.0, N))
        Phit = np.cumsum(hits * weights) / np.sum(weights[hits])
    Pmiss = np.cumsum(no_hits) / (N - Nh)
    ES = np.max(Phit - Pmiss)
    ES_neg = np.max(Pmiss - Phit)
    return ES if ES > ES_neg else -ES
def run_custom_wtcs(expr_matrix, up_genes, down_genes):
    wtcs_results = []
    for drug in expr_matrix.columns:
        expr = expr_matrix[drug].dropna()
        ranked_gene_ids = expr.sort_values(ascending=False).index.tolist()
        es_up = compute_gsea_score(ranked_gene_ids, set(up_genes))
        es_down = compute_gsea_score(ranked_gene_ids, set(down_genes))
        wtcs = (es_up - es_down) / 2
        wtcs_results.append({"drug_id": drug,"es_up": es_up,"es_down": es_down,"wtcs": wtcs})
    df = pd.DataFrame(wtcs_results).sort_values(by="wtcs").reset_index(drop=True)
    df["reverse_rank"] = df["wtcs"].rank(method="min", ascending=True).astype(int)
    percentile_ranks = rankdata(df["wtcs"], method="min") / len(df)
    tau_scores = (percentile_ranks * 200) - 100
    df["tau"] = tau_scores
    return df.sort_values(by="tau", ascending=True).reset_index(drop=True)
def filter_drug_rules(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Unified Drug Name'] = df['Drug'].apply(lambda x: x.split('.')[0])
    df_sorted = df.sort_values(by=['Unified Drug Name', 'Genes Matched'], ascending=[True, False])
    max_genes_matched = df_sorted.groupby('Unified Drug Name')['Genes Matched'].max().reset_index()
    max_genes_matched.columns = ['Unified Drug Name', 'Max Genes Matched']
    df_merged = pd.merge(df_sorted, max_genes_matched, on='Unified Drug Name')
    df_final = df_merged[df_merged['Genes Matched'] == df_merged['Max Genes Matched']]
    df_final.to_csv(output_file, index=False)
    return output_file
def run_ranking_analysis(input_file, output_file):
    df = pd.read_csv(input_file)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='drug_id', y='tau', data=df)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    df.to_csv(output_file, index=False)
    messagebox.showinfo("完成", f"药物排名分析完成，结果已保存至: {output_file}")
def configure_styles():
    style = ttk.Style()
    style.configure('TNotebook', background='#f0f0f0')
    style.configure('TNotebook.Tab', padding=[20, 8], font=('微软雅黑', 10))
    style.configure('Card.TFrame', background='white', relief='raised', borderwidth=1)
    style.configure('Modern.TButton', font=('微软雅黑', 10), padding=[10, 8])
    style.configure('Heading.TLabel', font=('微软雅黑', 12, 'bold'), background='white')
    style.configure('Info.TLabel', font=('微软雅黑', 9), background='white', foreground='#666666')
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("药物分析系统 - 登录")
        self.geometry("400x550")
        self.configure(bg='#2c3e50')
        self.resizable(False, False)
        self.center_window()
        main_frame = tk.Frame(self, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=30, pady=30)
        login_card = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        login_card.pack(expand=True, fill='both')
        title_frame = tk.Frame(login_card, bg='white')
        title_frame.pack(fill='x', pady=(25, 15))
        icon_label = tk.Label(title_frame, text="💊", font=("Arial", 32), bg='white', fg='#3498db')
        icon_label.pack()
        title_label = tk.Label(title_frame, text="基于扰动谱的靶向药物筛选系统", font=("微软雅黑", 14, "bold"), bg='white', fg='#2c3e50')
        title_label.pack(pady=(10, 0))
        subtitle_label = tk.Label(title_frame, text="Drug Screening System", font=("Arial", 10), bg='white', fg='#7f8c8d')
        subtitle_label.pack(pady=(5, 0))
        input_frame = tk.Frame(login_card, bg='white')
        input_frame.pack(fill='x', padx=30, pady=15)
        username_container = tk.Frame(input_frame, bg='white')
        username_container.pack(fill='x', pady=(0, 15))
        username_label = tk.Label(username_container, text="用户名", font=("微软雅黑", 10), bg='white', fg='#2c3e50')
        username_label.pack(anchor='w', pady=(0, 5))
        self.entry_username = tk.Entry(username_container, font=("微软雅黑", 11), relief='solid', bd=1, highlightthickness=1)
        self.entry_username.pack(fill='x', ipady=8)
        self.entry_username.bind("<FocusIn>", self.on_entry_focus_in)
        self.entry_username.bind("<FocusOut>", self.on_entry_focus_out)
        self.entry_username.bind("<Tab>", self.focus_password)
        password_container = tk.Frame(input_frame, bg='white')
        password_container.pack(fill='x', pady=(0, 20))
        password_label = tk.Label(password_container, text="密码", font=("微软雅黑", 10), bg='white', fg='#2c3e50')
        password_label.pack(anchor='w', pady=(0, 5))
        self.entry_password = tk.Entry(password_container, show="*", font=("微软雅黑", 11), relief='solid', bd=1, highlightthickness=1)
        self.entry_password.pack(fill='x', ipady=8)
        self.entry_password.bind("<FocusIn>", self.on_entry_focus_in)
        self.entry_password.bind("<FocusOut>", self.on_entry_focus_out)
        self.entry_password.bind("<Return>", lambda e: self.login())
        self.entry_password.bind("<Button-1>", self.password_click)
        button_container = tk.Frame(input_frame, bg='white')
        button_container.pack(fill='x', pady=(10, 0))
        self.login_btn = tk.Button(button_container, text="登录系统", command=self.login, font=("微软雅黑", 11, "bold"), bg='#3498db', fg='white', relief='flat', cursor='hand2', pady=10)
        self.login_btn.pack(fill='x')
        self.login_btn.bind("<Enter>", lambda e: self.login_btn.config(bg='#2980b9'))
        self.login_btn.bind("<Leave>", lambda e: self.login_btn.config(bg='#3498db'))
        tip_label = tk.Label(login_card, text="默认用户名和密码均为：1", font=("微软雅黑", 9), bg='white', fg='#95a5a6')
        tip_label.pack(pady=(5, 15))
        self.update_idletasks()
        self.after(100, lambda: self.entry_username.focus_set())
    def center_window(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.winfo_screenheight() // 2) - (550 // 2)
        self.geometry(f"400x550+{x}+{y}")
    def on_entry_focus_in(self, event):
        event.widget.config(highlightcolor='#3498db', highlightbackground='#3498db')
    def on_entry_focus_out(self, event):
        event.widget.config(highlightcolor='#bdc3c7', highlightbackground='#bdc3c7')
    def focus_password(self, event):
        self.entry_password.focus_set()
        return "break"
    def password_click(self, event):
        self.entry_password.focus_set()
    def login(self):
        username = self.entry_username.get()
        password = self.entry_password.get()
        if username == "1" and password == "1":
            self.withdraw()
            app = Application()
            app.protocol("WM_DELETE_WINDOW", self.on_main_close)
            app.mainloop()
            self.destroy()
        else:
            messagebox.showerror("登录失败", "用户名或密码错误，请重试！")
            self.entry_password.delete(0, tk.END)
            self.entry_username.focus_set()
    def on_main_close(self):
        self.quit()
        self.destroy()
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("基于扰动谱的靶向药物筛选系统")
        self.geometry("1000x700")
        self.configure(bg='#ecf0f1')
        self.center_window()
        configure_styles()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        main_container = tk.Frame(self, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_header(main_container)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True, pady=(10, 0))
        self.create_tabs()
    def on_closing(self):
        self.destroy()
        self.quit()
    def center_window(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.winfo_screenheight() // 2) - (700 // 2)
        self.geometry(f"1000x700+{x}+{y}")
    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg='#34495e', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        title_label = tk.Label(header_frame, text="💊 基于扰动谱的靶向药物筛选系统", font=("微软雅黑", 16, "bold"), bg='#34495e', fg='white')
        title_label.pack(side='left', padx=20, pady=15)
        version_label = tk.Label(header_frame, text="v1.0", font=("Arial", 10), bg='#34495e', fg='#bdc3c7')
        version_label.pack(side='right', padx=20, pady=15)
    def create_tabs(self):
        self.tab_data_input = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data_input, text="📊 数据输入与检查")
        self.tab_drug_regulation_ranking = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_drug_regulation_ranking, text="🔬 药物调控排名")
        self.tab_about_system = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_about_system, text="ℹ️ 关于系统")
        self.create_data_input_tab()
        self.create_drug_regulation_ranking_tab()
        self.create_about_system_tab()
    def create_data_input_tab(self):
        canvas = tk.Canvas(self.tab_data_input, bg='#ecf0f1')
        scrollbar = ttk.Scrollbar(self.tab_data_input, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content_frame = tk.Frame(scrollable_frame, bg='#ecf0f1')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        self.create_file_card(content_frame, "药物基因表达数据文件", "drug_data", 0)
        self.create_file_card(content_frame, "上调基因集文件", "up_genes", 1)
        self.create_file_card(content_frame, "下调基因集文件", "down_genes", 2)
        btn_frame = tk.Frame(content_frame, bg='#ecf0f1')
        btn_frame.pack(fill='x', pady=20)
        self.btn_confirm = tk.Button(btn_frame, text="✓ 确认文件选择", command=self.confirm_file_selection, font=("微软雅黑", 12, "bold"), bg='#27ae60', fg='white', relief='flat', cursor='hand2', pady=12, width=20)
        self.btn_confirm.pack()
        self.btn_confirm.bind("<Enter>", lambda e: self.btn_confirm.config(bg='#229954'))
        self.btn_confirm.bind("<Leave>", lambda e: self.btn_confirm.config(bg='#27ae60'))
    def create_file_card(self, parent, title, file_type, row):
        card_frame = tk.Frame(parent, bg='white', relief='raised', bd=1)
        card_frame.pack(fill='x', pady=10)
        content = tk.Frame(card_frame, bg='white')
        content.pack(fill='both', expand=True, padx=20, pady=15)
        title_label = tk.Label(content, text=f"📄 {title}", font=("微软雅黑", 12, "bold"), bg='white', fg='#2c3e50')
        title_label.pack(anchor='w', pady=(0, 10))
        file_frame = tk.Frame(content, bg='white')
        file_frame.pack(fill='x', pady=(0, 10))
        path_entry = tk.Entry(file_frame, font=("微软雅黑", 10), relief='solid', bd=1, state='readonly')
        path_entry.pack(side='left', fill='x', expand=True, ipady=5)
        browse_btn = tk.Button(file_frame, text="浏览文件", font=("微软雅黑", 10), bg='#3498db', fg='white', relief='flat', cursor='hand2', padx=15, pady=5)
        browse_btn.pack(side='right', padx=(10, 0))
        browse_btn.bind("<Enter>", lambda e: browse_btn.config(bg='#2980b9'))
        browse_btn.bind("<Leave>", lambda e: browse_btn.config(bg='#3498db'))
        preview_label = tk.Label(content, text="文件预览：", font=("微软雅黑", 10), bg='white', fg='#7f8c8d')
        preview_label.pack(anchor='w', pady=(10, 5))
        preview_text = tk.Text(content, height=4, font=("Consolas", 9), relief='solid', bd=1, bg='#f8f9fa', wrap=tk.NONE)
        preview_text.pack(fill='x')
        setattr(self, f'{file_type}_path_entry', path_entry)
        setattr(self, f'{file_type}_preview', preview_text)
        if file_type == 'drug_data':
            browse_btn.config(command=self.browse_wide_table)
        elif file_type == 'up_genes':
            browse_btn.config(command=self.browse_up_genes)
        elif file_type == 'down_genes':
            browse_btn.config(command=self.browse_down_genes)
    def browse_wide_table(self):
        self.wide_table_path = filedialog.askopenfilename(title="选择药物基因表达数据文件", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.wide_table_path:
            self.drug_data_path_entry.config(state='normal')
            self.drug_data_path_entry.delete(0, tk.END)
            self.drug_data_path_entry.insert(0, self.wide_table_path)
            self.drug_data_path_entry.config(state='readonly')
            self.preview_file(self.wide_table_path, "drug_data")
    def browse_up_genes(self):
        self.up_genes_path = filedialog.askopenfilename(title="选择上调基因集文件", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.up_genes_path:
            self.up_genes_path_entry.config(state='normal')
            self.up_genes_path_entry.delete(0, tk.END)
            self.up_genes_path_entry.insert(0, self.up_genes_path)
            self.up_genes_path_entry.config(state='readonly')
            self.preview_file(self.up_genes_path, "up_genes")
    def browse_down_genes(self):
        self.down_genes_path = filedialog.askopenfilename(title="选择下调基因集文件", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.down_genes_path:
            self.down_genes_path_entry.config(state='normal')
            self.down_genes_path_entry.delete(0, tk.END)
            self.down_genes_path_entry.insert(0, self.down_genes_path)
            self.down_genes_path_entry.config(state='readonly')
            self.preview_file(self.down_genes_path, "down_genes")
    def preview_file(self, file_path, file_type):
        try:
            df = pd.read_csv(file_path)
            preview = df.head(3).to_string(index=False)
            preview_widget = getattr(self, f'{file_type}_preview')
            preview_widget.delete(1.0, tk.END)
            preview_widget.insert(tk.END, preview)
        except Exception as e:
            messagebox.showerror("预览失败", f"无法预览文件: {str(e)}")
    def confirm_file_selection(self):
        if (hasattr(self, 'wide_table_path') and hasattr(self, 'up_genes_path') and hasattr(self, 'down_genes_path')):
            messagebox.showinfo("文件确认", "✓ 已确认所有输入文件，可以进行药物调控排名分析！")
            self.notebook.select(1)
        else:
            messagebox.showerror("文件选择不完整", "请先选择所有必需的输入文件")
    def create_drug_regulation_ranking_tab(self):
        main_frame = tk.Frame(self.tab_drug_regulation_ranking, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        analysis_card = tk.Frame(main_frame, bg='white', relief='raised', bd=1)
        analysis_card.pack(fill='x', pady=(0, 20))
        card_content = tk.Frame(analysis_card, bg='white')
        card_content.pack(fill='both', expand=True, padx=30, pady=25)
        title_label = tk.Label(card_content, text="🔬 药物调控排名分析", font=("微软雅黑", 14, "bold"), bg='white', fg='#2c3e50')
        title_label.pack(anchor='w', pady=(0, 15))
        output_frame = tk.Frame(card_content, bg='white')
        output_frame.pack(fill='x', pady=(0, 20))
        tk.Label(output_frame, text="选择结果保存路径：", font=("微软雅黑", 11), bg='white', fg='#34495e').pack(anchor='w', pady=(0, 8))
        path_frame = tk.Frame(output_frame, bg='white')
        path_frame.pack(fill='x')
        self.output_path_entry = tk.Entry(path_frame, font=("微软雅黑", 10), relief='solid', bd=1, state='readonly')
        self.output_path_entry.pack(side='left', fill='x', expand=True, ipady=6)
        output_browse_btn = tk.Button(path_frame, text="选择保存位置", command=self.browse_output_drug, font=("微软雅黑", 10), bg='#e67e22', fg='white', relief='flat', cursor='hand2', padx=15)
        output_browse_btn.pack(side='right', padx=(10, 0))
        output_browse_btn.bind("<Enter>", lambda e: output_browse_btn.config(bg='#d35400'))
        output_browse_btn.bind("<Leave>", lambda e: output_browse_btn.config(bg='#e67e22'))
        run_frame = tk.Frame(card_content, bg='white')
        run_frame.pack(fill='x', pady=(20, 0))
        self.btn_run_analysis = tk.Button(run_frame, text="🚀 开始药物调控分析", command=self.run_drug_analysis, font=("微软雅黑", 12, "bold"), bg='#8e44ad', fg='white', relief='flat', cursor='hand2', pady=15, width=25)
        self.btn_run_analysis.pack()
        self.btn_run_analysis.bind("<Enter>", lambda e: self.btn_run_analysis.config(bg='#7d3c98'))
        self.btn_run_analysis.bind("<Leave>", lambda e: self.btn_run_analysis.config(bg='#8e44ad'))
        self.progress_label = tk.Label(card_content, text="", font=("微软雅黑", 10), bg='white', fg='#7f8c8d')
        self.progress_label.pack(pady=(10, 0))
    def browse_output_drug(self):
        self.output_file_drug = filedialog.asksaveasfilename(title="选择结果保存位置", defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.output_file_drug:
            self.output_path_entry.config(state='normal')
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, self.output_file_drug)
            self.output_path_entry.config(state='readonly')
    def run_drug_analysis(self):
        if not all([hasattr(self, 'wide_table_path'), hasattr(self, 'up_genes_path'), hasattr(self, 'down_genes_path'), hasattr(self, 'output_file_drug')]):
            messagebox.showerror("分析失败", "请先完成所有文件的选择！")
            return
        try:
            self.progress_label.config(text="正在加载数据...")
            self.update()
            up_genes, down_genes = load_gene_set_from_symbol_files(self.up_genes_path, self.down_genes_path)
            expr_matrix = load_expression_matrix_from_wide_table(self.wide_table_path)
            if expr_matrix is None:
                self.progress_label.config(text="")
                messagebox.showerror("分析失败", "药物基因表达数据加载失败，请检查文件格式。")
                return
            self.progress_label.config(text="正在进行WTCS分析...")
            self.update()
            results_df = run_custom_wtcs(expr_matrix, up_genes, down_genes)
            results_df.to_csv(self.output_file_drug, index=False)
            self.progress_label.config(text="分析完成！正在生成图表...")
            self.update()
            plt.style.use('seaborn-v0_8')
            plt.figure(figsize=(12, 7))
            sns.barplot(x='drug_id', y='tau', data=results_df.head(20), palette='viridis')
            plt.title('Top 20 (Tau Score)', fontsize=14, fontweight='bold')
            plt.xlabel('Drug ID', fontsize=12)
            plt.ylabel('Tau Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            self.progress_label.config(text="")
            messagebox.showinfo("分析完成", f"✓ 药物调控排名分析完成！\n\n结果已保存至:\n{self.output_file_drug}\n\n共分析了 {len(results_df)} 个药物")
        except Exception as e:
            self.progress_label.config(text="")
            messagebox.showerror("分析失败", f"分析过程中出现错误:\n{str(e)}")
    def create_about_system_tab(self):
        canvas = tk.Canvas(self.tab_about_system, bg='#ecf0f1')
        scrollbar = ttk.Scrollbar(self.tab_about_system, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content_frame = tk.Frame(scrollable_frame, bg='#ecf0f1')
        content_frame.pack(fill='both', expand=True, padx=30, pady=30)
        intro_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        intro_card.pack(fill='x', pady=(0, 20))
        intro_content = tk.Frame(intro_card, bg='white')
        intro_content.pack(fill='both', expand=True, padx=25, pady=20)
        tk.Label(intro_content, text="💊 系统介绍", font=("微软雅黑", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))
        intro_text = """本系统以收集的药物转录谱或扰动谱为基础，通过分析关键疾病靶点与药物谱之间的调节方向和调节幅度，快速筛选出数据库中药物对所选靶点的调控排名。
系统采用WTCS (Weighted Total Connectivity Score) 算法，结合GSEA (Gene Set Enrichment Analysis) 方法，为药物研发和精准医疗提供有力支持。"""
        tk.Label(intro_content, text=intro_text, font=("微软雅黑", 11), bg='white', fg='#34495e', justify='left', wraplength=800).pack(anchor='w')
        faq_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        faq_card.pack(fill='x', pady=(0, 20))
        faq_content = tk.Frame(faq_card, bg='white')
        faq_content.pack(fill='both', expand=True, padx=25, pady=20)
        tk.Label(faq_content, text="❓ 常见问题", font=("微软雅黑", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))
        faqs = [("如何选择正确的基因集文件？", "用户需要确保基因集文件格式正确，并且包含上下调基因的基因ID信息。文件应为CSV格式，并且必须包含名为'gene_id'的列。"),
                ("如何查看分析结果？", "分析结果会在界面中展示图表，同时保存为CSV文件。用户可以在指定位置找到详细的分析结果数据。"),
                ("如何修改输入数据格式？", "确保数据文件是标准的CSV格式，并且列名与预期一致。药物表达数据需要包含'DrugName'列作为索引。")]
        for i, (question, answer) in enumerate(faqs):
            q_frame = tk.Frame(faq_content, bg='white')
            q_frame.pack(fill='x', pady=(0 if i == 0 else 15, 0))
            tk.Label(q_frame, text=f"Q{i+1}: {question}", font=("微软雅黑", 11, "bold"), bg='white', fg='#3498db').pack(anchor='w')
            tk.Label(q_frame, text=f"A: {answer}", font=("微软雅黑", 10), bg='white', fg='#34495e', justify='left', wraplength=800).pack(anchor='w', pady=(5, 0))
        contact_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        contact_card.pack(fill='x')
        contact_content = tk.Frame(contact_card, bg='white')
        contact_content.pack(fill='both', expand=True, padx=25, pady=20)
        tk.Label(contact_content, text="📧 联系方式", font=("微软雅黑", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))
        tk.Label(contact_content, text="如有问题或建议，请联系：19834920338@163.com", font=("微软雅黑", 11), bg='white', fg='#34495e').pack(anchor='w')
        tk.Label(contact_content, text="版本：v1.0 | 更新日期：2025", font=("微软雅黑", 9), bg='white', fg='#95a5a6').pack(anchor='w', pady=(10, 0))
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    login_window = LoginWindow()
    login_window.mainloop()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use('TkAgg')


def load_expression_matrix_from_wide_table(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    if 'DrugName' in df.columns:
        df.set_index('DrugName', inplace=True)
    else:
        messagebox.showerror("Error", "No column named 'DrugName' found, please check the data format.")
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
        wtcs_results.append({"drug_id": drug, "es_up": es_up, "es_down": es_down, "wtcs": wtcs})
    df = pd.DataFrame(wtcs_results).sort_values(by="wtcs").reset_index(drop=True)
    df["reverse_rank"] = df["wtcs"].rank(method="min", ascending=True).astype(int)
    percentile_ranks = rankdata(df["wtcs"], method="min") / len(df)
    tau_scores = (percentile_ranks * 200) - 100
    df["tau"] = tau_scores
    return df.sort_values(by="tau", ascending=True).reset_index(drop=True)


def configure_styles():
    style = ttk.Style()
    style.configure('TNotebook', background='#f0f0f0')
    style.configure('TNotebook.Tab', padding=[20, 8], font=('Microsoft YaHei', 10))
    style.configure('Card.TFrame', background='white', relief='raised', borderwidth=1)
    style.configure('Modern.TButton', font=('Microsoft YaHei', 10), padding=[10, 8])
    style.configure('Heading.TLabel', font=('Microsoft YaHei', 12, 'bold'), background='white')
    style.configure('Info.TLabel', font=('Microsoft YaHei', 9), background='white', foreground='#666666')


class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drug Analysis System - Login")
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

        title_label = tk.Label(title_frame, text="Targeted Drug Screening System",
                               font=("Microsoft YaHei", 14, "bold"), bg='white', fg='#2c3e50')
        title_label.pack(pady=(10, 0))

        subtitle_label = tk.Label(title_frame, text="Based on Perturbation Spectra",
                                  font=("Arial", 10), bg='white', fg='#7f8c8d')
        subtitle_label.pack(pady=(5, 0))

        input_frame = tk.Frame(login_card, bg='white')
        input_frame.pack(fill='x', padx=30, pady=15)

        username_container = tk.Frame(input_frame, bg='white')
        username_container.pack(fill='x', pady=(0, 15))
        username_label = tk.Label(username_container, text="Username",
                                  font=("Microsoft YaHei", 10), bg='white', fg='#2c3e50')
        username_label.pack(anchor='w', pady=(0, 5))
        self.entry_username = tk.Entry(username_container, font=("Microsoft YaHei", 11),
                                       relief='solid', bd=1, highlightthickness=1)
        self.entry_username.pack(fill='x', ipady=8)
        self.entry_username.bind("<FocusIn>", self.on_entry_focus_in)
        self.entry_username.bind("<FocusOut>", self.on_entry_focus_out)

        password_container = tk.Frame(input_frame, bg='white')
        password_container.pack(fill='x', pady=(0, 20))
        password_label = tk.Label(password_container, text="Password",
                                  font=("Microsoft YaHei", 10), bg='white', fg='#2c3e50')
        password_label.pack(anchor='w', pady=(0, 5))
        self.entry_password = tk.Entry(password_container, show="*", font=("Microsoft YaHei", 11),
                                       relief='solid', bd=1, highlightthickness=1)
        self.entry_password.pack(fill='x', ipady=8)
        self.entry_password.bind("<FocusIn>", self.on_entry_focus_in)
        self.entry_password.bind("<FocusOut>", self.on_entry_focus_out)
        self.entry_password.bind("<Return>", lambda e: self.login())

        button_container = tk.Frame(input_frame, bg='white')
        button_container.pack(fill='x', pady=(10, 0))

        self.login_btn = tk.Button(button_container, text="Login to System", command=self.login,
                                   font=("Microsoft YaHei", 11, "bold"), bg='#3498db', fg='white',
                                   relief='flat', cursor='hand2', pady=10)
        self.login_btn.pack(fill='x')
        self.login_btn.bind("<Enter>", lambda e: self.login_btn.config(bg='#2980b9'))
        self.login_btn.bind("<Leave>", lambda e: self.login_btn.config(bg='#3498db'))

        tip_label = tk.Label(login_card, text="Default username and password are both: 1",
                             font=("Microsoft YaHei", 9), bg='white', fg='#95a5a6')
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

    def login(self):
        username = self.entry_username.get()
        password = self.entry_password.get()
        if username == "1" and password == "1":
            self.destroy()
            app = Application()
            app.mainloop()
        else:
            messagebox.showerror("Login Failed", "Username or password incorrect, please try again!")
            self.entry_password.delete(0, tk.END)
            self.entry_username.focus_set()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Targeted Drug Screening System")
        self.geometry("1400x800")
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
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.destroy()

    def center_window(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.winfo_screenheight() // 2) - (800 // 2)
        self.geometry(f"1400x800+{x}+{y}")

    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg='#34495e', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame,
                               text="💊 Targeted Drug Screening System Based on Perturbation Spectra",
                               font=("Microsoft YaHei", 16, "bold"), bg='#34495e', fg='white')
        title_label.pack(side='left', padx=20, pady=15)

        version_label = tk.Label(header_frame, text="v1.0",
                                 font=("Arial", 10), bg='#34495e', fg='#bdc3c7')
        version_label.pack(side='right', padx=20, pady=15)

    def create_tabs(self):
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text=" Drug Regulation Analysis")

        self.tab_about_system = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_about_system, text=" About System")

        self.create_analysis_tab()
        self.create_about_system_tab()

    def create_analysis_tab(self):
        # 主容器 - 左右分栏
        main_container = tk.Frame(self.tab_analysis, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)

        # 左侧面板 - 数据输入 (40%)
        left_panel = tk.Frame(main_container, bg='#ecf0f1')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 8))

        # 右侧面板 - 分析和结果 (60%)
        right_panel = tk.Frame(main_container, bg='#ecf0f1')
        right_panel.pack(side='right', fill='both', expand=True, padx=(8, 0))

        # === 左侧：数据输入区域 ===
        self.create_left_panel(left_panel)

        # === 右侧：分析和结果区域 ===
        self.create_right_panel(right_panel)

    def create_left_panel(self, parent):
        # Step 1: 数据输入卡片
        input_card = tk.Frame(parent, bg='white', relief='raised', bd=1)
        input_card.pack(fill='both', expand=True)

        input_content = tk.Frame(input_card, bg='white')
        input_content.pack(fill='both', expand=True, padx=20, pady=20)

        # 标题
        title_frame = tk.Frame(input_content, bg='white')
        title_frame.pack(fill='x', pady=(0, 15))

        tk.Label(title_frame, text=" Data Input",
                 font=("Microsoft YaHei", 13, "bold"), bg='white', fg='#2c3e50').pack(side='left')

        self.status_indicator = tk.Label(title_frame, text="⭕ Pending",
                                         font=("Microsoft YaHei", 9), bg='white', fg='#95a5a6')
        self.status_indicator.pack(side='right')

        # 文件输入区域
        self.create_file_input(input_content, "Drug Gene Expression Data", "drug_data",
                               "CSV file containing drug expression profiles")
        self.create_file_input(input_content, "Upregulated Gene Set", "up_genes",
                               "CSV file with 'gene_id' column")
        self.create_file_input(input_content, "Downregulated Gene Set", "down_genes",
                               "CSV file with 'gene_id' column")

        # 文件信息摘要
        separator = tk.Frame(input_content, bg='#e0e0e0', height=1)
        separator.pack(fill='x', pady=15)

        tk.Label(input_content, text="📈 Data Summary",
                 font=("Microsoft YaHei", 11, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 10))

        summary_frame = tk.Frame(input_content, bg='#f8f9fa', relief='solid', bd=1)
        summary_frame.pack(fill='x', pady=(0, 10))

        summary_content = tk.Frame(summary_frame, bg='#f8f9fa')
        summary_content.pack(fill='both', expand=True, padx=15, pady=10)

        self.summary_labels = {}
        summary_items = [
            ("Drugs:", "drugs", "0"),
            ("Up Genes:", "up_genes_count", "0"),
            ("Down Genes:", "down_genes_count", "0"),
            ("Status:", "file_status", "No files loaded")
        ]

        for label, key, default in summary_items:
            row = tk.Frame(summary_content, bg='#f8f9fa')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, font=("Microsoft YaHei", 9),
                     bg='#f8f9fa', fg='#555').pack(side='left')
            self.summary_labels[key] = tk.Label(row, text=default,
                                                font=("Microsoft YaHei", 9, "bold"),
                                                bg='#f8f9fa', fg='#2c3e50')
            self.summary_labels[key].pack(side='right')

    def create_file_input(self, parent, label_text, file_type, description):
        container = tk.Frame(parent, bg='white')
        container.pack(fill='x', pady=(0, 15))

        # 标题行
        title_row = tk.Frame(container, bg='white')
        title_row.pack(fill='x', pady=(0, 5))

        tk.Label(title_row, text=f"📄 {label_text}",
                 font=("Microsoft YaHei", 10, "bold"), bg='white', fg='#34495e').pack(side='left')

        status_label = tk.Label(title_row, text="", font=("Microsoft YaHei", 8), bg='white')
        status_label.pack(side='right')
        setattr(self, f'{file_type}_status', status_label)

        # 描述
        tk.Label(container, text=description, font=("Microsoft YaHei", 8),
                 bg='white', fg='#7f8c8d').pack(anchor='w', pady=(0, 5))

        # 文件选择
        file_row = tk.Frame(container, bg='white')
        file_row.pack(fill='x', pady=(0, 5))

        path_entry = tk.Entry(file_row, font=("Microsoft YaHei", 9),
                              relief='solid', bd=1, state='readonly', fg='#555555')
        path_entry.pack(side='left', fill='x', expand=True, ipady=4)

        browse_btn = tk.Button(file_row, text="Browse",
                               font=("Microsoft YaHei", 9),
                               cursor='hand2', padx=15, pady=4)
        browse_btn.pack(side='right', padx=(8, 0))

        # 预览
        preview_text = tk.Text(container, height=2, font=("Consolas", 8),
                               relief='solid', bd=1, bg='#f8f9fa', wrap=tk.NONE)
        preview_text.pack(fill='x')

        setattr(self, f'{file_type}_path_entry', path_entry)
        setattr(self, f'{file_type}_preview', preview_text)

        if file_type == 'drug_data':
            browse_btn.config(command=self.browse_wide_table)
        elif file_type == 'up_genes':
            browse_btn.config(command=self.browse_up_genes)
        elif file_type == 'down_genes':
            browse_btn.config(command=self.browse_down_genes)

    def create_right_panel(self, parent):
        # Step 2: 分析配置
        config_card = tk.Frame(parent, bg='white', relief='raised', bd=1)
        config_card.pack(fill='x', pady=(0, 10))

        config_content = tk.Frame(config_card, bg='white')
        config_content.pack(fill='both', expand=True, padx=20, pady=15)

        tk.Label(config_content, text=" Analysis Configuration",
                 font=("Microsoft YaHei", 13, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))

        # 输出路径
        tk.Label(config_content, text="Output File Path:",
                 font=("Microsoft YaHei", 10), bg='white', fg='#34495e').pack(anchor='w', pady=(0, 5))

        output_row = tk.Frame(config_content, bg='white')
        output_row.pack(fill='x', pady=(0, 10))

        self.output_path_entry = tk.Entry(output_row, font=("Microsoft YaHei", 9),
                                          relief='solid', bd=1, state='readonly')
        self.output_path_entry.pack(side='left', fill='x', expand=True, ipady=5)

        output_btn = tk.Button(output_row, text="Browse", command=self.browse_output_drug,
                               font=("Microsoft YaHei", 9),
                               cursor='hand2', padx=15)
        output_btn.pack(side='right', padx=(10, 0))

        # 参数设置
        params_frame = tk.Frame(config_content, bg='white')
        params_frame.pack(fill='x', pady=(10, 0))

        # Top N显示
        top_n_row = tk.Frame(params_frame, bg='white')
        top_n_row.pack(fill='x', pady=(0, 8))

        tk.Label(top_n_row, text="Display Top N Drugs:",
                 font=("Microsoft YaHei", 9), bg='white', fg='#34495e').pack(side='left')

        self.top_n_var = tk.StringVar(value="20")
        top_n_spinbox = tk.Spinbox(top_n_row, from_=5, to=100, increment=5,
                                   textvariable=self.top_n_var, width=10,
                                   font=("Microsoft YaHei", 9))
        top_n_spinbox.pack(side='right')

        # Step 3: 执行和结果
        run_card = tk.Frame(parent, bg='white', relief='raised', bd=1)
        run_card.pack(fill='both', expand=True)

        run_content = tk.Frame(run_card, bg='white')
        run_content.pack(fill='both', expand=True, padx=20, pady=15)

        tk.Label(run_content, text=" Run Analysis & Results",
                 font=("Microsoft YaHei", 13, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))

        # 运行按钮容器
        button_container = tk.Frame(run_content, bg='white')
        button_container.pack(fill='x', pady=(0, 15))

        self.btn_run_analysis = tk.Button(button_container, text="▶  Start Analysis",
                                          command=self.run_drug_analysis,
                                          font=("Microsoft YaHei", 11, "bold"),
                                          cursor='hand2', pady=12, padx=20)
        self.btn_run_analysis.pack(fill='x')

        # 进度条容器
        progress_container = tk.Frame(run_content, bg='white')
        progress_container.pack(fill='x', pady=(0, 15))

        # 进度标签
        progress_header = tk.Frame(progress_container, bg='white')
        progress_header.pack(fill='x', pady=(0, 5))

        tk.Label(progress_header, text="Analysis Progress:",
                 font=("Microsoft YaHei", 9, "bold"), bg='white', fg='#34495e').pack(side='left')

        self.progress_percentage = tk.Label(progress_header, text="0%",
                                            font=("Microsoft YaHei", 9, "bold"),
                                            bg='white', fg='#2c3e50')
        self.progress_percentage.pack(side='right')

        # 进度条样式配置
        progress_style = ttk.Style()
        progress_style.theme_use('clam')  # 使用clam主题以确保自定义颜色生效
        progress_style.configure("Green.Horizontal.TProgressbar",
                                 troughcolor='#ecf0f1',
                                 bordercolor='#bdc3c7',
                                 background='#95A5A6',
                                 lightcolor='#95A5A6',
                                 darkcolor='#95A5A6',
                                 thickness=20)

        self.progress = ttk.Progressbar(progress_container,
                                        style="Green.Horizontal.TProgressbar",
                                        mode='determinate',
                                        maximum=100,
                                        length=400)
        self.progress.pack(fill='x', pady=(0, 5))

        self.progress_label = tk.Label(progress_container, text="Ready to start analysis",
                                       font=("Microsoft YaHei", 9), bg='white', fg='#7f8c8d')
        self.progress_label.pack()
        # 统计卡片
        stats_container = tk.Frame(run_content, bg='white')
        stats_container.pack(fill='x', pady=(0, 15))

        self.stat_cards = {}
        stat_items = [
            ("Total Drugs", "total"),
            ("Analysis Time", "time"),
            ("Max Tau", "max_tau"),
            ("Min Tau", "min_tau")
        ]

        for i, (label, key) in enumerate(stat_items):
            card = tk.Frame(stats_container, relief='solid', bd=1)
            card.pack(side='left', fill='both', expand=True, padx=(0 if i == 0 else 5, 0))

            card_content = tk.Frame(card)
            card_content.pack(fill='both', expand=True, padx=10, pady=8)

            tk.Label(card_content, text=label, font=("Microsoft YaHei", 8)).pack()

            value_label = tk.Label(card_content, text="-",
                                   font=("Microsoft YaHei", 14, "bold"))
            value_label.pack()
            self.stat_cards[key] = value_label

        # 结果表格
        tk.Label(run_content, text=" Top Results Preview",
                 font=("Microsoft YaHei", 11, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(15, 8))

        # 创建表格框架
        table_frame = tk.Frame(run_content, bg='white')
        table_frame.pack(fill='both', expand=True)

        # 创建Treeview
        columns = ('Rank', 'Drug ID', 'Tau Score', 'ES Up', 'ES Down')
        self.result_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)

        # 定义列
        self.result_table.heading('Rank', text='Rank')
        self.result_table.heading('Drug ID', text='Drug ID')
        self.result_table.heading('Tau Score', text='Tau Score')
        self.result_table.heading('ES Up', text='ES Up')
        self.result_table.heading('ES Down', text='ES Down')

        self.result_table.column('Rank', width=50, anchor='center')
        self.result_table.column('Drug ID', width=120, anchor='w')
        self.result_table.column('Tau Score', width=100, anchor='center')
        self.result_table.column('ES Up', width=80, anchor='center')
        self.result_table.column('ES Down', width=80, anchor='center')

        # 滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=scrollbar.set)

        self.result_table.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def update_file_status(self):
        files_loaded = sum([
            hasattr(self, 'wide_table_path'),
            hasattr(self, 'up_genes_path'),
            hasattr(self, 'down_genes_path')
        ])

        if files_loaded == 3:
            self.status_indicator.config(text="✓ Complete", fg='#27ae60')
            self.summary_labels['file_status'].config(text="All files loaded", fg='#27ae60')
        elif files_loaded > 0:
            self.status_indicator.config(text="⏳ In Progress", fg='#f39c12')
            self.summary_labels['file_status'].config(text=f"{files_loaded}/3 files loaded", fg='#f39c12')
        else:
            self.status_indicator.config(text="⭕ Pending", fg='#95a5a6')
            self.summary_labels['file_status'].config(text="No files loaded", fg='#95a5a6')

    def browse_wide_table(self):
        self.wide_table_path = filedialog.askopenfilename(
            title="Select Drug Gene Expression Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.wide_table_path:
            self.drug_data_path_entry.config(state='normal')
            self.drug_data_path_entry.delete(0, tk.END)
            self.drug_data_path_entry.insert(0, self.wide_table_path)
            self.drug_data_path_entry.config(state='readonly')
            self.preview_file(self.wide_table_path, "drug_data")
            self.drug_data_status.config(text="✓ Loaded", fg='#27ae60')

            # 更新摘要
            try:
                df = pd.read_csv(self.wide_table_path, low_memory=False)
                self.summary_labels['drugs'].config(text=str(len(df)))
            except:
                pass

            self.update_file_status()

    def browse_up_genes(self):
        self.up_genes_path = filedialog.askopenfilename(
            title="Select Upregulated Gene Set File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.up_genes_path:
            self.up_genes_path_entry.config(state='normal')
            self.up_genes_path_entry.delete(0, tk.END)
            self.up_genes_path_entry.insert(0, self.up_genes_path)
            self.up_genes_path_entry.config(state='readonly')
            self.preview_file(self.up_genes_path, "up_genes")
            self.up_genes_status.config(text="✓ Loaded", fg='#27ae60')

            try:
                df = pd.read_csv(self.up_genes_path)
                self.summary_labels['up_genes_count'].config(text=str(len(df)))
            except:
                pass

            self.update_file_status()

    def browse_down_genes(self):
        self.down_genes_path = filedialog.askopenfilename(
            title="Select Downregulated Gene Set File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.down_genes_path:
            self.down_genes_path_entry.config(state='normal')
            self.down_genes_path_entry.delete(0, tk.END)
            self.down_genes_path_entry.insert(0, self.down_genes_path)
            self.down_genes_path_entry.config(state='readonly')
            self.preview_file(self.down_genes_path, "down_genes")
            self.down_genes_status.config(text="✓ Loaded", fg='#27ae60')

            try:
                df = pd.read_csv(self.down_genes_path)
                self.summary_labels['down_genes_count'].config(text=str(len(df)))
            except:
                pass

            self.update_file_status()

    def preview_file(self, file_path, file_type):
        try:
            df = pd.read_csv(file_path)
            preview = df.head(2).to_string(index=False)
            preview_widget = getattr(self, f'{file_type}_preview')
            preview_widget.delete(1.0, tk.END)
            preview_widget.insert(tk.END, preview)
        except Exception as e:
            pass

    def browse_output_drug(self):
        self.output_file_drug = filedialog.asksaveasfilename(
            title="Select Result Save Location",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if self.output_file_drug:
            self.output_path_entry.config(state='normal')
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, self.output_file_drug)
            self.output_path_entry.config(state='readonly')

    def run_drug_analysis(self):
        # 强制刷新进度条样式
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar",
                        background='#95A5A6',
                        lightcolor='#95A5A6',
                        darkcolor='#95A5A6')
        if not all([hasattr(self, 'wide_table_path'),
                    hasattr(self, 'up_genes_path'),
                    hasattr(self, 'down_genes_path'),
                    hasattr(self, 'output_file_drug')]):
            messagebox.showerror("Analysis Failed", "Please complete all file selections first!")
            return

        try:
            import time
            start_time = time.time()

            # 禁用按钮并重置进度
            self.btn_run_analysis.config(state='disabled', text="⏳  Processing...")
            self.progress['value'] = 0
            self.progress_percentage.config(text="0%")
            self.progress_label.config(text="⏳ Initializing...", fg='#2c3e50')
            self.update()

            # 阶段1: 加载数据 (0-30%)
            self.update_progress(10, "📂 Loading gene sets...")
            up_genes, down_genes = load_gene_set_from_symbol_files(
                self.up_genes_path, self.down_genes_path)

            self.update_progress(20, "📊 Loading drug expression data...")
            expr_matrix = load_expression_matrix_from_wide_table(self.wide_table_path)

            if expr_matrix is None:
                self.reset_analysis_ui()
                self.progress_label.config(text="❌ Error: Failed to load data", fg='#e74c3c')
                return

            self.update_progress(30, "✓ Data loaded successfully")

            # 阶段2: 执行分析 (30-90%)
            self.progress_label.config(text="🔬 Performing WTCS analysis...")
            self.update()

            total_drugs = len(expr_matrix.columns)
            results_list = []

            for idx, drug in enumerate(expr_matrix.columns):
                # 计算进度
                progress = 30 + int((idx / total_drugs) * 60)
                self.update_progress(progress, f"🔬 Analyzing drug {idx + 1}/{total_drugs}: {drug}")

                expr = expr_matrix[drug].dropna()
                ranked_gene_ids = expr.sort_values(ascending=False).index.tolist()
                es_up = compute_gsea_score(ranked_gene_ids, set(up_genes))
                es_down = compute_gsea_score(ranked_gene_ids, set(down_genes))
                wtcs = (es_up - es_down) / 2
                results_list.append({"drug_id": drug, "es_up": es_up, "es_down": es_down, "wtcs": wtcs})

            # 阶段3: 后处理 (90-100%)
            self.update_progress(90, "📊 Processing results...")

            results_df = pd.DataFrame(results_list).sort_values(by="wtcs").reset_index(drop=True)
            results_df["reverse_rank"] = results_df["wtcs"].rank(method="min", ascending=True).astype(int)
            percentile_ranks = rankdata(results_df["wtcs"], method="min") / len(results_df)
            tau_scores = (percentile_ranks * 200) - 100
            results_df["tau"] = tau_scores
            results_df = results_df.sort_values(by="tau", ascending=True).reset_index(drop=True)

            self.update_progress(95, "💾 Saving results...")
            results_df.to_csv(self.output_file_drug, index=False)

            end_time = time.time()
            analysis_time = end_time - start_time

            # 完成
            self.update_progress(100, "✓ Analysis completed successfully!")
            self.progress_label.config(fg='#27ae60')
            self.reset_analysis_ui()

            # 更新统计卡片
            self.stat_cards['total'].config(text=str(len(results_df)))
            self.stat_cards['time'].config(text=f"{analysis_time:.1f}s")
            self.stat_cards['max_tau'].config(text=f"{results_df['tau'].max():.2f}")
            self.stat_cards['min_tau'].config(text=f"{results_df['tau'].min():.2f}")

            # 更新结果表格
            self.result_table.delete(*self.result_table.get_children())
            top_n = int(self.top_n_var.get())
            for idx, row in results_df.head(top_n).iterrows():
                self.result_table.insert('', 'end', values=(
                    idx + 1,
                    row['drug_id'],
                    f"{row['tau']:.2f}",
                    f"{row['es_up']:.3f}",
                    f"{row['es_down']:.3f}"
                ))

            # 显示图表
            self.show_result_chart(results_df)

            messagebox.showinfo("Analysis Completed",
                                f"✓ Analysis completed!\n\n" +
                                f"Total drugs analyzed: {len(results_df)}\n" +
                                f"Time taken: {analysis_time:.1f} seconds\n" +
                                f"Results saved to: {self.output_file_drug.split('/')[-1]}")

        except Exception as e:
            self.reset_analysis_ui()
            self.progress_label.config(text="❌ Error during analysis", fg='#e74c3c')
            messagebox.showerror("Analysis Failed", f"Error: {str(e)}")

    def update_progress(self, value, message):
        """更新进度条和消息"""
        self.progress['value'] = value
        self.progress_percentage.config(text=f"{int(value)}%")
        self.progress_label.config(text=message)
        self.update()

    def reset_analysis_ui(self):
        """重置分析界面"""
        self.btn_run_analysis.config(state='normal', text="▶  Start Analysis")

    def show_result_chart(self, results_df):
        top_n = int(self.top_n_var.get())

        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 6))

        top_data = results_df.head(top_n)
        bars = ax.barh(range(len(top_data)), top_data['tau'], color='#3498db')

        ax.set_yticks(range(len(top_data)))
        ax.set_yticklabels(top_data['drug_id'])
        ax.set_xlabel('Tau Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Drug ID', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Drugs by Tau Score', fontsize=13, fontweight='bold')
        ax.invert_yaxis()

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f' {width:.2f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def create_about_system_tab(self):
        canvas = tk.Canvas(self.tab_about_system, bg='#ecf0f1')
        scrollbar = ttk.Scrollbar(self.tab_about_system, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')

        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content_frame = tk.Frame(scrollable_frame, bg='#ecf0f1')
        content_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # 系统介绍
        intro_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        intro_card.pack(fill='x', pady=(0, 20))

        intro_content = tk.Frame(intro_card, bg='white')
        intro_content.pack(fill='both', expand=True, padx=25, pady=20)

        tk.Label(intro_content, text="💊 System Introduction",
                 font=("Microsoft YaHei", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))

        intro_text = """This system is based on collected drug transcriptomic or perturbation spectra, by analyzing the regulation direction and amplitude between key disease targets and drug spectra, quickly screens the regulation ranking of drugs in the database for the selected targets.

The system uses WTCS (Weighted Total Connectivity Score) algorithm, combined with GSEA (Gene Set Enrichment Analysis) method, to provide strong support for drug development and precision medicine."""

        tk.Label(intro_content, text=intro_text, font=("Microsoft YaHei", 11),
                 bg='white', fg='#34495e', justify='left', wraplength=800).pack(anchor='w')

        # 常见问题
        faq_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        faq_card.pack(fill='x', pady=(0, 20))

        faq_content = tk.Frame(faq_card, bg='white')
        faq_content.pack(fill='both', expand=True, padx=25, pady=20)

        tk.Label(faq_content, text="❓ Frequently Asked Questions",
                 font=("Microsoft YaHei", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))

        faqs = [
            ("How to select the correct gene set file?",
             "Users need to ensure the gene set file format is correct and contains gene ID information for up and down regulated genes. The file should be in CSV format and must include a column named 'gene_id'."),
            ("How to view analysis results?",
             "Analysis results will be displayed in charts in the interface, and saved as CSV files. Users can find detailed analysis result data at the specified location."),
            ("How to modify input data format?",
             "Ensure the data file is in standard CSV format, and column names match expectations. Drug expression data needs to include 'DrugName' column as index.")
        ]

        for i, (question, answer) in enumerate(faqs):
            q_frame = tk.Frame(faq_content, bg='white')
            q_frame.pack(fill='x', pady=(0 if i == 0 else 15, 0))

            tk.Label(q_frame, text=f"Q{i + 1}: {question}",
                     font=("Microsoft YaHei", 11, "bold"), bg='white', fg='#3498db').pack(anchor='w')

            tk.Label(q_frame, text=f"A: {answer}",
                     font=("Microsoft YaHei", 10), bg='white', fg='#34495e',
                     justify='left', wraplength=800).pack(anchor='w', pady=(5, 0))

        # 联系信息
        contact_card = tk.Frame(content_frame, bg='white', relief='raised', bd=1)
        contact_card.pack(fill='x')

        contact_content = tk.Frame(contact_card, bg='white')
        contact_content.pack(fill='both', expand=True, padx=25, pady=20)

        tk.Label(contact_content, text="📧 Contact Information",
                 font=("Microsoft YaHei", 14, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 15))

        tk.Label(contact_content, text="If you have any questions or suggestions, please contact: 19834920338@163.com",
                 font=("Microsoft YaHei", 11), bg='white', fg='#34495e').pack(anchor='w')

        tk.Label(contact_content, text="Version: v1.0 | Update Date: 2025",
                 font=("Microsoft YaHei", 9), bg='white', fg='#95a5a6').pack(anchor='w', pady=(10, 0))


if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        login_window = LoginWindow()
        login_window.mainloop()
    except Exception as e:
        print(f"程序启动错误: {e}")
        import traceback

        traceback.print_exc()
        input("按回车键退出...")
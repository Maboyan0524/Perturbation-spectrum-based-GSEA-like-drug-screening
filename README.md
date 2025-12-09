#  Targeted Drug Screening System

A desktop application for targeted drug screening based on perturbation spectra using WTCS and GSEA algorithms.


### Login Interface

Default username and password: `1`

<img width="339" height="488" alt="ed350e705e743b310054e6aad5ee49e4" src="https://github.com/user-attachments/assets/e7bbe663-62f1-4bc5-b62c-45044a98b1db" />

### Main Interface - Analysis Workflow

<img width="1920" height="1032" alt="be7b2ab800cc0508ca25830621bcebe6" src="https://github.com/user-attachments/assets/b0b2f19d-db97-4137-a7ce-a07d0f8eb212" />

Complete workflow: Data Input → Configuration → Analysis → Results，Real-time progress tracking and comprehensive results visualization

### About system
 <img width="1920" height="1032" alt="834a519f200108b7973c8b88cd4eaec5" src="https://github.com/user-attachments/assets/a258055e-fde7-4208-b5c9-f302fd3b75cd" />
Notes and feedback

##  Key Features

-  **WTCS Algorithm**: Weighted Total Connectivity Score for drug-target analysis
-  **GSEA Integration**: Gene Set Enrichment Analysis for statistical evaluation
-  **Intuitive Interface**: Step-by-step workflow with progress tracking
-  **Visualization**: Automatic ranking and statistical summary
-  **Batch Processing**: Analyze multiple drugs simultaneously
-  **Editable**: Fully customizable gene sets to adapt to personal data processing
##  Installation & Usage

### Requirements

- Python 3.7 or higher
- Windows / macOS / Linux
                                        

### Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 2: Run Application

```bash
python system.py
```

### Step 3: Login

- Username: `1`
- Password: `1`

### Step 4: Analyze Data

1. **Load Files**: Upload drug expression data and gene sets (CSV format)
2. **Configure**: Set output path and display parameters
3. **Run**: Click "Start Analysis" button
4. **View Results**: Check statistics, rankings, and visualizations

##  Input Data Format

### Drug Expression Data (CSV)
```csv
DrugName,GENE1,GENE2,GENE3,...
Drug_A,2.5,1.3,-0.8,...
Drug_B,-1.2,3.4,0.5,...
```
- First column must be `DrugName`
- Numeric expression values

### Gene Sets (CSV)
```csv
gene_id
GENE1
GENE2
GENE3
```
- Column name must be `gene_id`
- One gene per row

##  Understanding Results

**Tau Score Interpretation:**
- **Negative values (< 0)**: Drug reverses disease pattern → Potential therapeutic effect
- **Positive values (> 0)**: Drug mimics disease pattern → Avoid

**Output Includes:**
- Ranked drug list with Tau scores
- Enrichment scores (ES Up / ES Down)
- Statistical summary (Total drugs, Analysis time, Max/Min Tau)
- Visualization charts

##  Contact
**Version**: v1.0  
**Update Date**: 2025
                                      

##  License

MIT License

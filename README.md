# 🎯 Smadex Creative Intelligence Platform

An intelligent ad campaign analysis and optimization platform powered by machine learning. This project extracts, analyzes, and optimizes creative assets for advertising campaigns using Gemini AI for data extraction and XGBoost for performance prediction.

---

## 🚀 Features

- **Campaign Data Extraction**: Uses Gemini AI to extract and complete campaign metadata from user descriptions
- **Creative Analysis**: Analyzes up to 6 creative variations per campaign
- **Smart Imputation**: Intelligent handling of missing data using feature importance and correlation analysis
- **Performance Prediction**: XGBoost models predict key performance indicators (CTR, CVR, IPM, ROAS, CPA)
- **Fatigue Detection**: Identifies when creatives start to lose effectiveness over time
- **Interactive Dashboard**: Streamlit-based UI for campaign configuration and results visualization

---

## 📁 Project Structure

```
main_repository/
├── app.py                    # Main Streamlit application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Dataset files (CSV)
│   ├── creative_summary.csv  # Pre-aggregated creative metrics
│   ├── creative_daily_country_os_stats.csv  # Daily performance data
│   ├── advertisers.csv       # Advertiser metadata
│   ├── campaigns.csv         # Campaign setup data
│   ├── creatives.csv         # Creative metadata
│   └── data_dictionary.csv   # Column definitions
├── src/                      # Backend source code
│   ├── gemini_ai_service.py # Gemini AI integration for data extraction
│   ├── hack_en.py           # XGBoost ML pipeline and fatigue detection
│   ├── styles.py            # Custom UI styling
│   └── utils/
│       └── ranges.json      # Configuration ranges
└── vistes/                   # View layer
    └── chatbot.py           # Main chatbot interface
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| AI/ML | Google Gemini, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Streamlit native components |

---

## ⚙️ Installation

1. **Clone the repository**

2. **Create your virtual environment**

```python3 -m venv venv```

3. **Start your environment**

```source venv/bin/activate```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file with your Gemini API key:**

```
GEMINI_API_TOKEN=your_api_key_here
```
You can generate this API key in [here](https://aistudio.google.com/welcome?utm_source=google&utm_medium=cpc&utm_campaign=Cloud-SS-DR-AIS-FY26-global-gsem-1713578&utm_content=text-ad&utm_term=KW_api%20key%20gemini&gclsrc=aw.ds&gad_source=1&gad_campaignid=23417416058&gclid=CjwKCAjwzLHPBhBTEiwABaLsSlkfoXigZmIBUUNFW7BQvjcunHqnjPjYZv-u8CeibpMrcJj3N_kqgBoCSDgQAvD_BwE)

---

## 🎯 Usage

1. **Run the application:**

```bash
streamlit run app.py
```

2. **Click "Start Analysis"** on the home page

3. **Enter campaign context** and select target KPI

4. **Upload up to 6 creative variations** (descriptions and images)

5. **Click "Analyze Campaign"** to extract data

6. **Review extracted data** and click "Optimize Campaign"

7. **View predictions** and performance analysis

---

## 📊 Supported KPIs

| KPI | Description |
|-----|-------------|
| **CTR** | Click Through Rate - Ad attractiveness (clicks) |
| **CVR** | Conversion Rate - Conversion funnel effectiveness |
| **IPM** | Installs Per Mille - Gaming effectiveness (installs/1k) |
| **ROAS** | Return on Ad Spend - Economic return on investment |
| **CPA** | Cost Per Action - Cost per sale/registration |

---

## 🔧 Key Modules

### `src/gemini_ai_service.py`
- Extracts campaign and creative data from user inputs using Gemini AI
- Validates vertical consistency across creatives
- Returns structured JSON with all metadata

### `src/hack_en.py`
- `compute_fatigue()`: Detects when creatives lose effectiveness based on CTR/CPA trends
- `smart_nan_imputation()`: Fills missing values using importance-weighted correlations
- `run_xgboost_pipeline()`: Main ML pipeline for optimization

### `vistes/chatbot.py`
- Multi-step wizard interface for campaign analysis
- Displays predictions, comparisons, and export options

---

## 📈 Dataset

The project uses a synthetic AdTech dataset with:

- 36 advertisers
- 180 campaigns
- 1,080 creatives
- 192,315 daily performance records

### Join Keys

```
advertisers.advertiser_id = campaigns.advertiser_id
campaigns.campaign_id = creatives.campaign_id
creatives.creative_id = creative_daily_country_os_stats.creative_id
campaigns.campaign_id = creative_daily_country_os_stats.campaign_id
```

See [data/README.md](data/README.md) for detailed dataset documentation.

---

## 📝 License

This project is created for the Smadex Hackathon challenge.
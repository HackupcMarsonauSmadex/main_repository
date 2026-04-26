import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Main app background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #f9fafb 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            border-right: 1px solid #e5e7eb;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        h1 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-size: 1.6rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        h3 {
            font-size: 1.2rem;
            margin-top: 1.2rem;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        }

        /* Containers */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            background: white;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }

        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-color: #d1d5db;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f0f4f8 0%, #f8fafc 100%);
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-weight: 600;
            color: #1e293b;
            padding: 12px 16px;
        }

        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #e2e8f0 0%, #f1f5f9 100%);
            border-color: #cbd5e1;
        }

        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            background-color: #f9fafb;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
            background-color: white;
        }

        /* Metrics */
        .stMetric {
            background: white;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .stMetricLabel {
            color: #64748b;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .stMetricValue {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        /* Divider */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
            margin: 1.5rem 0;
        }

        /* Alerts */
        [data-testid="stAlertContainer"] {
            border-radius: 8px;
            border-left: 4px solid;
            padding: 12px 16px;
        }

        /* Caption and text */
        .st-caption, small {
            color: #64748b;
            font-weight: 500;
        }

        /* Data table */
        [data-testid="dataframe"] {
            border-radius: 8px;
            overflow: hidden;
        }

        th {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            font-weight: 700;
            padding: 10px 12px;
        }

        /* Radio buttons */
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #f8f9fa 0%, #f0f4f8 100%);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 6px;
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        div[role="radiogroup"] label:hover {
            border-color: #cbd5e1;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
            color: #1e40af;
            border-color: #3b82f6;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
            font-weight: 600;
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.2);
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
            box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
        }

        /* Spinner */
        .stSpinner > div {
            border-color: rgba(59, 130, 246, 0.2);
            border-right-color: #3b82f6;
        }

        /* Animation */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(5px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            animation: fadeIn 0.3s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)
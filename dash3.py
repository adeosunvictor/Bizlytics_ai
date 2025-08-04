import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sweetviz
import tempfile
import os
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from google.generativeai import configure, GenerativeModel
from prophet import Prophet 
from fpdf import FPDF
import textwrap
import base64
import uuid
import re


configure(api_key=st.secrets["GOOGLE_API_KEY"])


gemini_model = GenerativeModel("gemini-2.5-pro")

# Initialize session state variables
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'saved_forecasts' not in st.session_state:
    st.session_state.saved_forecasts = []
if 'saved_charts' not in st.session_state:
    st.session_state.saved_charts = []
if 'ai_summary' not in st.session_state:
    st.session_state.ai_summary = ""
if 'cleaning_done' not in st.session_state:
    st.session_state.cleaning_done = False

from PIL import Image
import streamlit as st


# ui
import streamlit as st
# Replace the entire UI section with this properly formatted version:

# UI Section
st.set_page_config(page_title="Bizlytics", layout="wide")

# 📸 Hero Image
st.image("hero.png", use_column_width=True)

# ✨ Welcome Message
st.markdown("## ✨ Your AI Business Analyst — On Demand")
st.markdown("""
**Bizlytics helps you unlock powerful business insights** from raw data — instantly.  
No code. No tools. Just upload your spreadsheet and go from confusion to clarity.
""")

# 💡 Key Capabilities
st.markdown("### 💡 Key Capabilities")
st.markdown("""
- 🔍 **Auto-cleans** and understands your data  
- 📊 **Visualizes** revenue, cost, and profit trends  
- ⚠️ **Detects** outliers and hidden anomalies using AI  
- 📅 **Forecasts** future performance with time series modeling  
- 🧠 **Generates** an executive summary with smart insights  
- 📄 **Exports** a professional, shareable PDF report  
""")

st.markdown("""
Whether you're a **founder**, **marketer**, or **finance lead**,  
Bizlytics turns your spreadsheets into strategic superpowers.
""")

# 📊 Insight Screenshot
st.image("insight1.png", caption="AI-Powered Insights", use_column_width=True)

# 📁 Supported Data
st.markdown("### 📁 What Kind of Data Can You Upload?")
st.markdown("""
Upload a `.csv` file with:
- ✅ Sales or revenue reports  
- ✅ Marketing performance data  
- ✅ Financial statements (e.g. revenue, cost, profit)  
- ✅ Customer transactions or product orders  
- ✅ Time-series metrics with date columns  
""")

st.markdown("⚠️ **Heads-up:** Make sure your dataset is clean:")
st.markdown("""
- ✔ Column names are present and descriptive  
- ✔ No merged or empty headers  
- ✔ Consistent data types (e.g. no mixing text & numbers in one column)  
""")
st.info("A clean dataset = accurate insights. Garbage in, garbage out.")




# 📂 Upload CSV
uploaded_file = st.file_uploader("Upload your business data file (CSV)", type=["csv"])


# 📬 Contact Section
st.markdown("""
---  
### 📬 Contact Us

Want to partner, integrate, or just say hello?

- 🧑‍💼 **Name:** Adeosun Victor  
- 📧 **Email:** [victoradeosun15@gmail.com](mailto:victoradeosun15@gmail.com)  
- 📞 **Phone:** +234 907 562 8094  
- 🔗 **LinkedIn:** [linkedin.com/in/adeosunvictor](https://linkedin.com/in/adeosunvictor)
""")

# Footer
st.markdown(
    "<hr><div style='text-align: center; padding-top: 10px; font-size: 0.85rem;'>"
    "© 2025 <strong>Bizlytics</strong>. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.cleaned_df = df.copy()
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # 📊 EDA Report
        if st.button("Generate EDA Report"):
            with st.spinner("Generating EDA report..."):
                report = sweetviz.analyze(df)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    report.show_html(tmp.name, open_browser=False)
                    with open(tmp.name, "r") as f:
                        html = f.read()
                    st.components.v1.html(html, height=800, scrolling=True)
                os.unlink(tmp.name)
        
        # 📈 KPI Dashboard
        st.header("📈 Financial Summary Dashboard")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Revenue' in numeric_cols and 'Cost' in numeric_cols:
            df['Profit'] = df['Revenue'] - df['Cost']
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Revenue", f"£{df['Revenue'].sum():,.2f}")
            col2.metric("Total Cost", f"£{df['Cost'].sum():,.2f}")
            col3.metric("Total Profit", f"£{df['Profit'].sum():,.2f}")

            fig = px.line(df, x=df.columns[0], y=['Revenue', 'Cost', 'Profit'], 
                         title="Revenue, Cost & Profit Trends")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.saved_charts.append((fig, "Revenue, Cost & Profit Trends"))

        # Sweetviz inline preview (Cross-platform compatible)
        st.subheader("🧠 In-App Sweetviz Preview")
        if not st.session_state.cleaning_done:
            report = sweetviz.analyze(st.session_state.cleaned_df)
            
            # Create a temporary directory that works on all platforms
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"sweetviz_{uuid.uuid4()}.html")
            
            report.show_html(temp_path, open_browser=False)
            
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)
            finally:
                # Clean up the temporary file and directory
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    st.warning(f"Could not clean up temporary files: {e}")
            
            st.session_state.cleaning_done = True
        else:
            st.info("✅ Already processed. You can proceed to visualization and forecasting.")

        # 📉 Visualizations
        st.subheader("📈 Visual Insights")
        if numeric_cols:
            selected = st.selectbox("Select a numeric column", numeric_cols)
            fig = px.histogram(df, x=selected, title=f"Distribution of {selected}")
            st.plotly_chart(fig)
            st.session_state.saved_charts.append((fig, f"Distribution of {selected}"))

        # 🚨 Improved Anomaly Detection
        st.subheader("🚨 Anomaly Detection")
        if numeric_cols:
            anomaly_col = st.selectbox("Select column to detect anomalies", numeric_cols)
            if anomaly_col:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[[anomaly_col]])
                
                model_iforest = IsolationForest(
                    contamination=0.05,
                    n_estimators=200,
                    max_samples='auto',
                    random_state=42
                )
                
                df["anomaly"] = model_iforest.fit_predict(scaled_values)
                anomalies = df[df["anomaly"] == -1]
                
                st.write(f"Detected {len(anomalies)} anomalies ({(len(anomalies)/len(df)*100):.2f}% of data)")
                
                fig = px.scatter(
                    df, 
                    x=df.index, 
                    y=anomaly_col,
                    color=df["anomaly"].map({1: "Normal", -1: "Anomaly"}),
                    title=f"Anomaly Detection in {anomaly_col}",
                    labels={"color": "Status"}
                )
                st.plotly_chart(fig)
                st.dataframe(anomalies)

        # Forecasting
        # Forecasting Section with Improved Visualization
        st.header("📅 Forecast Future Trends")
        date_columns = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        if date_columns and numeric_cols:
            date_col = st.selectbox("Select Date Column", date_columns)
            value_col = st.selectbox("Select Value Column to Forecast", numeric_cols)

            # Convert and validate datetime column
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            df_forecast = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

            prophet_model = Prophet()
            prophet_model.fit(df_forecast)

            periods_input = st.slider("Forecast Horizon (days)", 30, 730, 365)
            future = prophet_model.make_future_dataframe(periods=periods_input)
            forecast = prophet_model.predict(future)

            # Forecast Plot with Customized Size and Legend
            st.subheader("📉 Forecast Plot")
            st.markdown("""
            **Visual Elements Explanation:**
            - **Black Dots**: Actual observed values from your historical data
            - **Blue Line**: Prophet's forecast/prediction
            - **Light Blue Area**: Uncertainty interval (80% confidence by default)
            """)
            
            fig1 = prophet_model.plot(forecast)
            # Adjust the figure size
            fig1.set_size_inches(10, 5)  # Width, Height in inches
            st.pyplot(fig1, clear_figure=True)  # Clear figure to prevent overlap
            
            # Forecast Components with Smaller Frame
            st.subheader("🔍 Forecast Components")
            st.markdown("""
            **Component Breakdown:**
            - **Trend**: Overall direction of the metric
            - **Weekly**: Repeating weekly patterns
            - **Yearly**: Annual seasonality (if enough data)
            """)
            
            fig2 = prophet_model.plot_components(forecast)
            fig2.set_size_inches(10, 8)  # Adjust component plot size
            st.pyplot(fig2, clear_figure=True)
            
            st.session_state.saved_forecasts.append((fig1, f"Forecast for {value_col}"))

        # 🤖 AI Summary
        # 🤖 AI Summary Section (fixed version)
        st.subheader("🧠 AI Executive Summary")
        if st.button("Generate AI Summary", key="ai_summary_button"):  # Added unique key
            with st.spinner("Generating AI summary..."):
                forecast_titles = [title for _, title in st.session_state.saved_forecasts]
                chart_titles = [title for _, title in st.session_state.saved_charts]
                sample = df.head(10).to_dict(orient="records")
                columns = ", ".join(df.columns)

                # Define prompt here (was missing)
                prompt = f"""
        You are an expert AI business analyst. Provide a clear, concise analysis with these formatting rules:
        1. Use bullet points for lists
        2. Avoid excessive commas and semicolons
        3. Keep paragraphs short and scannable
        4. Use bold for key metrics
        5. Format percentages as 'X%' without decimals unless needed

        Dataset Columns: {columns}
        Sample rows: {sample}
        Charts: {chart_titles}
        Forecasts: {forecast_titles}

        Tasks:
        1. Describe current performance
        2. Compare key columns (sales vs discount, profit vs quantity)
        3. Forecast expected business performance 30 days ahead
        4. Provide formulas used for insights
        5. Give 3 specific actionable insights grounded in the data

        Be realistic and data-driven. Avoid filler words and unnecessary punctuation.
        """
                try:
                    response = gemini_model.generate_content(prompt)
                    # Clean up the response text
                    clean_text = response.text.replace(";", ".")  # Replace semicolons with periods
                    clean_text = clean_text.replace(",,", ",")    # Fix double commas
                    clean_text = "\n".join([line.strip() for line in clean_text.split("\n") if line.strip()])  # Remove empty lines
                    
                    st.session_state.ai_summary = clean_text
                    st.markdown("### 📋 Executive Summary")
                    st.markdown(clean_text)  # Using markdown for better formatting
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

        # PDF Export Section (fixed with proper indentation)
        st.subheader("📄 Download AI Summary Report")
        if st.session_state.ai_summary:
            class StyledPDF(FPDF):
                def __init__(self):
                    super().__init__()
                    self.logo = "logo.png"
                    self.logo_width = 30
                    self.has_logo = os.path.exists(self.logo)

                def header(self):
                    if self.has_logo:
                        try:
                            self.image(self.logo, x=(210 - self.logo_width)/2, y=10, w=self.logo_width)
                            self.set_y(40)
                        except:
                            self.set_y(20)
                    else:
                        self.set_y(20)
                    
                    self.set_font('Arial', 'B', 16)
                    self.cell(0, 10, 'Bizlytics Analytical Report', 0, 1, 'C')
                    self.ln(10)

                def section_title(self, title):  # THIS IS THE FIX - ADDED MISSING METHOD
                    self.set_font('Arial', 'B', 14)
                    self.cell(0, 10, title, 0, 1)
                    self.ln(5)

                def clean_text(self, text):
                    text = re.sub(r'[#*]', '', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = re.sub(r' {2,}', ' ', text)
                    return text.strip()

                def section_body(self, text):
                    self.set_font('Arial', '', 11)
                    cleaned_text = self.clean_text(text)
                    
                    paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip()]
                    for para in paragraphs:
                        if para.startswith("-"):
                            self.set_font('', 'B')
                            self.cell(8)
                            self.multi_cell(0, 6, para[1:].strip())
                            self.set_font('', '')
                        elif para.upper() == para and len(para) < 50:
                            self.set_font('', 'B')
                            self.multi_cell(0, 8, para)
                            self.set_font('', '')
                        else:
                            self.multi_cell(0, 6, para)
                        self.ln(4)

            if st.button("Generate and Download PDF Report", key="pdf_download_button"):
                pdf = StyledPDF()
                pdf.add_page()
                
                # Safely clean AI summary
                clean_text = st.session_state.ai_summary.encode('latin-1', 'ignore').decode('latin-1')
                pdf.section_title("BIZLYTICS REPORT")
                pdf.section_body(clean_text)

                try:
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    b64 = base64.b64encode(pdf_bytes).decode()

                    href = f'<a href="data:application/pdf;base64,{b64}" download="Analytical_Report.pdf" '\
                        'style="display: inline-block; background: #4CAF50; color: white; '\
                        'padding: 8px 16px; text-decoration: none; border-radius: 4px; '\
                        'margin-top: 10px;">Download PDF Report</a>'

                    st.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")

        else:
            st.warning("Please generate an AI summary first")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("📂 Upload a CSV file to get started.")
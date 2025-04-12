"""
Financial reports component for the PSX dashboard.
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
import base64
import io
import re
import requests
from urllib.parse import urlparse, unquote
import mimetypes
import tempfile
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Hugging Face API
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
# Change to a more accessible model
MODEL_ENDPOINTS = {
    "primary": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "backup": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
    "fallback": "https://api-inference.huggingface.co/models/google/flan-t5-large"
}

# Optional imports with fallback
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    if not hasattr(st.session_state, 'showed_pdf_warning'):
        st.warning("""
        PDF support is not available. To enable PDF analysis, please install PyPDF2:
        ```
        pip install PyPDF2
        ```
        """)
        st.session_state.showed_pdf_warning = True

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    if not hasattr(st.session_state, 'showed_docx_warning'):
        st.warning("""
        DOCX support is not available. To enable DOCX analysis, please install python-docx:
        ```
        pip install python-docx
        ```
        """)
        st.session_state.showed_docx_warning = True

# Add version lock at the top of the file
VERSION = "1.0.0"
VERSION_LOCKED = True

def check_version_lock():
    """Silently check if the version is locked without affecting functionality."""
    return VERSION_LOCKED

def get_config_path():
    """Get the path to the configuration file."""
    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "api_keys.json")

def load_api_keys():
    """Load API keys from configuration file."""
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                keys = json.load(f)
        except Exception as e:
            st.warning(f"Error loading API keys: {str(e)}")
            keys = {}
    else:
        keys = {}
    
    # Get DeepSeek API key from .env as fallback
    deepseek_key = os.getenv('DEEPSEEK_API_KEY', '')
    
    if deepseek_key and (not keys.get('deepseek') or keys.get('deepseek') == ''):
        keys['deepseek'] = deepseek_key
    
    # Ensure all keys exist in the dictionary
    for key_type in ['huggingface', 'deepseek', 'anthropic', 'openai', 'google', 'xai']:
        if key_type not in keys:
            keys[key_type] = ''
    
    return keys

def save_api_keys(keys):
    """Save API keys to configuration file."""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(keys, f)
    except Exception as e:
        st.error(f"Error saving API keys: {str(e)}")

def load_announcements_data(file_path: str):
    """
    Load PSX announcements from Excel file.
    
    Args:
        file_path: Path to the Excel file containing PSX announcements
        
    Returns:
        DataFrame containing the announcements data
    """
    try:
        df = pd.read_excel(file_path)
        # Convert date columns to datetime if they exist
        date_columns = ['Date', 'Announcement_Date', 'Publication_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading announcements data: {str(e)}")
        return None

def filter_announcements(df: pd.DataFrame, 
                         company: str = None, 
                         category: str = None,
                         start_date: datetime = None,
                         end_date: datetime = None,
                         search_term: str = None):
    """
    Filter announcements based on criteria.
    
    Args:
        df: DataFrame containing announcements
        company: Company symbol to filter by
        category: Announcement category to filter by
        start_date: Start date for filtering
        end_date: End date for filtering
        search_term: Text to search for in announcement content
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply filters if they are provided
    if company and 'Symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Symbol'] == company]
    
    if category and 'Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Category'] == category]
    
    # Apply date filters
    date_col = next((col for col in ['Date', 'Announcement_Date', 'Publication_Date'] 
                     if col in filtered_df.columns), None)
    
    if date_col and start_date:
        # Convert date to datetime if it's a date object
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        
        # Ensure the column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        
        filtered_df = filtered_df[filtered_df[date_col] >= start_date]
    
    if date_col and end_date:
        # Convert date to datetime if it's a date object
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())
        
        # Ensure the column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        
        filtered_df = filtered_df[filtered_df[date_col] <= end_date]
    
    # Apply text search if provided
    if search_term and search_term.strip():
        search_term = search_term.lower()
        text_columns = ['Details', 'Announcement', 'Description', 'Content', 'Title', 'Subject']
        
        # Create a mask for rows that match the search term
        mask = pd.Series(False, index=filtered_df.index)
        for col in text_columns:
            if col in filtered_df.columns:
                # Convert to string and handle NaN values
                col_values = filtered_df[col].fillna('').astype(str)
                mask = mask | col_values.str.lower().str.contains(search_term, na=False)
        
        filtered_df = filtered_df[mask]
    
    return filtered_df

def extract_file_extension(file_path: str) -> str:
    """Extract file extension from a file path."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def is_pdf_file(file_path: str) -> bool:
    """Check if a file is a PDF."""
    return extract_file_extension(file_path) == '.pdf'

def is_docx_file(file_path: str) -> bool:
    """Check if a file is a DOCX."""
    return extract_file_extension(file_path) == '.docx'

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    if not PDF_SUPPORT:
        return "PDF text extraction not available. Please install PyPDF2."
    
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    if not DOCX_SUPPORT:
        return "DOCX text extraction not available. Please install python-docx."
    
    try:
        docx_file = io.BytesIO(docx_bytes)
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def extract_text_from_file(file_bytes: bytes, file_path: str) -> str:
    """Extract text from a file based on its extension."""
    if is_pdf_file(file_path):
        return extract_text_from_pdf(file_bytes)
    elif is_docx_file(file_path):
        return extract_text_from_docx(file_bytes)
    else:
        return "Text extraction not supported for this file type."

def get_file_download_link(file_path: str, link_text: str) -> str:
    """Generate a download link for a file."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        ext = extract_file_extension(file_path)
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    except Exception as e:
        return f"Error generating download link: {str(e)}"

def display_file_preview(file_path: str, file_name: str = None):
    """Display a preview of a file if possible."""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    # Get file extension
    ext = extract_file_extension(file_path)
    
    # Create a container for the preview
    preview_container = st.container()
    
    with preview_container:
        # Create header with file info and download button
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(f"### 📄 {file_name}")
        with header_col2:
            download_link = get_file_download_link(file_path, "📥 Download")
            st.markdown(download_link, unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("---")
        
        # Try to extract and display text content
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            text_content = extract_text_from_file(file_bytes, file_path)
            
            # Format the text content
            formatted_content = format_preview_content(text_content)
            
            # Create tabs for different view options
            preview_tab, raw_tab = st.tabs(["📋 Formatted View", "📝 Raw Text"])
            
            with preview_tab:
                # Display formatted content in a custom container
                st.markdown(
                    f"""
                    <div style="
                        background-color: white;
                        padding: 20px;
                        border-radius: 5px;
                        border: 1px solid #ddd;
                        max-height: 600px;
                        overflow-y: auto;
                        font-family: Arial, sans-serif;
                        line-height: 1.5;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    ">
                        {formatted_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with raw_tab:
                # Display raw text in a text area for easy copying
                st.text_area(
                    "",
                    text_content,
                    height=600,
                    key=f"raw_text_{hash(file_path)}",
                    help="Raw text content that can be copied"
                )
            
        except Exception as e:
            st.error(f"Error previewing file: {str(e)}")

def format_preview_content(text: str) -> str:
    """Format the preview content for better readability."""
    try:
        # Escape HTML special characters
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Add paragraph breaks
        text = text.replace("\n\n", "</p><p>")
        text = f"<p>{text}</p>"
        
        # Format common financial report elements
        # Headers (all caps lines)
        text = re.sub(
            r'<p>([A-Z][A-Z\s\d.,;:\'\"&()/-]+)(\s*)(?=</p>)',
            r'<p><strong style="color: #2c5282;">\1</strong>\2',
            text
        )
        
        # Numbers and percentages
        text = re.sub(
            r'((?:Rs\.|PKR|USD)?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?)',
            r'<span style="color: #2b6cb0;">\1</span>',
            text
        )
        
        # Dates
        text = re.sub(
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'<span style="color: #4a5568;">\1</span>',
            text
        )
        
        # Important keywords
        keywords = [
            "profit", "loss", "revenue", "income", "expense", "dividend",
            "earnings", "share", "capital", "asset", "liability", "equity",
            "increase", "decrease", "growth", "decline"
        ]
        for keyword in keywords:
            pattern = re.compile(f'\\b{keyword}\\b', re.IGNORECASE)
            text = pattern.sub(f'<span style="color: #4c51bf;">{keyword}</span>', text)
        
        return text
    except Exception:
        # If any error occurs during formatting, return the original text
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_file(url: str) -> Optional[bytes]:
    """Download a file from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

def get_file_type(file_path: str) -> str:
    """Get the file type from the file path or URL."""
    # Get the extension
    ext = extract_file_extension(file_path)
    if not ext and is_valid_url(file_path):
        # Try to get content type from URL
        try:
            response = requests.head(file_path, timeout=5)
            content_type = response.headers.get('content-type', '')
            ext = mimetypes.guess_extension(content_type) or ''
        except:
            ext = ''
    
    # Map extensions to file types
    ext_map = {
        '.pdf': 'PDF',
        '.doc': 'Word Document',
        '.docx': 'Word Document',
        '.xls': 'Excel Spreadsheet',
        '.xlsx': 'Excel Spreadsheet',
        '.txt': 'Text File',
        '.csv': 'CSV File',
        '.html': 'HTML File',
        '.htm': 'HTML File'
    }
    
    return ext_map.get(ext.lower(), 'Unknown')

def create_temp_file(content: bytes, extension: str) -> str:
    """Create a temporary file with the given content and extension."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        st.error(f"Error creating temporary file: {str(e)}")
        return None

def display_report_preview(file_path: str, file_name: str = None):
    """Display a preview of a report file."""
    try:
        # Handle URLs
        temp_path = None
        if is_valid_url(file_path):
            with st.spinner("Downloading report..."):
                content = download_file(file_path)
                if content is None:
                    return
                
                # Create a temporary file
                ext = extract_file_extension(file_path)
                temp_path = create_temp_file(content, ext)
                if temp_path is None:
                    return
                
                file_path = temp_path
        
        # Display the file preview
        display_file_preview(file_path, file_name)
        
        # Clean up temporary file if it was created
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        st.error(f"Error previewing report: {str(e)}")

def get_report_url(row: pd.Series) -> Optional[str]:
    """Extract report URL from an announcement row."""
    # List of possible column names that might contain report URLs
    url_columns = ['URL', 'Link', 'Report_URL', 'File_URL', 'Attachment_URL', 'Document_URL']
    
    # Check each column
    for col in url_columns:
        if col in row.index and pd.notna(row[col]) and row[col]:
            url = str(row[col]).strip()
            if is_valid_url(url):
                return url
    
    # Check content columns for URLs
    content_columns = ['Details', 'Announcement', 'Description', 'Content']
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    for col in content_columns:
        if col in row.index and pd.notna(row[col]):
            content = str(row[col])
            urls = re.findall(url_pattern, content)
            for url in urls:
                if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                    return url
    
    return None

def display_announcement_details(row: pd.Series, date_col: str = None):
    """Display detailed information about an announcement."""
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display main content
        content_columns = ['Details', 'Announcement', 'Description', 'Content']
        for col in content_columns:
            if col in row.index and pd.notna(row[col]) and row[col]:
                st.markdown(f"**{col}:**")
                st.markdown(row[col])
    
    with col2:
        # Display metadata
        st.markdown("**Announcement Details:**")
        
        # Create a dictionary of metadata to display
        metadata = {}
        if 'Symbol' in row.index and pd.notna(row['Symbol']):
            metadata['Company'] = row['Symbol']
        if 'Category' in row.index and pd.notna(row['Category']):
            metadata['Category'] = row['Category']
        if 'Subject' in row.index and pd.notna(row['Subject']):
            metadata['Subject'] = row['Subject']
        if date_col and pd.notna(row[date_col]):
            metadata['Date'] = row[date_col].strftime('%Y-%m-%d %H:%M')
        
        # Display metadata
        for key, value in metadata.items():
            st.markdown(f"**{key}:** {value}")
        
        # Check for report URL
        report_url = get_report_url(row)
        if report_url:
            st.markdown("**Report:**")
            st.markdown(f'<a href="{report_url}" target="_blank" class="download-link">📥 Download Report</a>', unsafe_allow_html=True)
        
        # Check for attachments
        attachment_columns = [col for col in row.index if 'attachment' in col.lower() or 'file' in col.lower()]
        if attachment_columns:
            st.markdown("**Attachments:**")
            for col in attachment_columns:
                if pd.notna(row[col]) and row[col]:
                    value = str(row[col])
                    if is_valid_url(value):
                        st.markdown(f'<a href="{value}" target="_blank" class="download-link">📎 {col}</a>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"📎 {value}")

def display_financial_reports(config: Dict[str, Any]):
    """Display financial reports and announcements analysis."""
    # Silently check version lock without displaying messages
    check_version_lock()
    
    st.markdown("## 📑 Financial Reports & Announcements")
    
    with st.sidebar:
        st.markdown("### 🔑 AI Analysis Configuration")
        
        # Model selection
        st.markdown("#### 🤖 Select Analysis Model")
        selected_model = st.selectbox(
            "Choose Model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]['description']})" + (" 🔄" if MODEL_OPTIONS[x].get("beta_status", False) else ""),
            help="Select the AI model for analysis. Different models have different strengths."
        )
        
        # Store selected model in session state
        st.session_state.selected_model = selected_model
        
        # Show model strengths
        st.markdown("**Model Strengths:**")
        for strength in MODEL_OPTIONS[selected_model]['strengths']:
            st.markdown(f"- {strength}")
        
        # API Key configuration based on selected model
        model_info = MODEL_OPTIONS[selected_model]
        api_keys = load_api_keys()
        
        if model_info['api_type'] == "deepseek":
            st.markdown("#### 🔑 DeepSeek API Configuration")
            current_api_key = st.session_state.get('deepseek_api_key', api_keys.get('deepseek', ''))
            
            # Check if there's a key in .env
            env_key = os.getenv('DEEPSEEK_API_KEY', '')
            if env_key and (not current_api_key or current_api_key == ''):
                st.info("ℹ️ Using DeepSeek API key from environment variables")
                st.session_state.deepseek_api_key = env_key
                api_keys['deepseek'] = env_key
                save_api_keys(api_keys)
                current_api_key = env_key
            
            if current_api_key:
                st.success("✅ DeepSeek API key is already configured")
                if st.button("Change DeepSeek API Key", key="change_deepseek_key_sidebar"):
                    st.session_state.deepseek_api_key = ''
                    api_keys['deepseek'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "DeepSeek API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your DeepSeek API key",
                    key="deepseek_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.deepseek_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "deepseek")
                        if success:
                            st.success(message)
                            api_keys['deepseek'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "anthropic":
            st.markdown("#### 🔑 Anthropic API Configuration")
            current_api_key = st.session_state.get('anthropic_api_key', api_keys.get('anthropic', ''))
            
            if current_api_key:
                st.success("✅ Anthropic API key is already configured")
                if st.button("Change Anthropic API Key", key="change_anthropic_key_sidebar"):
                    st.session_state.anthropic_api_key = ''
                    api_keys['anthropic'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Anthropic API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Anthropic API key",
                    key="anthropic_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.anthropic_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "anthropic")
                        if success:
                            st.success(message)
                            api_keys['anthropic'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "openai":
            st.markdown("#### 🔑 OpenAI API Configuration")
            current_api_key = st.session_state.get('openai_api_key', api_keys.get('openai', ''))
            
            if current_api_key:
                st.success("✅ OpenAI API key is already configured")
                if st.button("Change OpenAI API Key", key="change_openai_key_sidebar"):
                    st.session_state.openai_api_key = ''
                    api_keys['openai'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "OpenAI API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your OpenAI API key",
                    key="openai_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.openai_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "openai")
                        if success:
                            st.success(message)
                            api_keys['openai'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "google":
            st.markdown("#### 🔑 Google API Configuration")
            current_api_key = st.session_state.get('google_api_key', api_keys.get('google', ''))
            
            if current_api_key:
                st.success("✅ Google API key is already configured")
                if st.button("Change Google API Key", key="change_google_key_sidebar"):
                    st.session_state.google_api_key = ''
                    api_keys['google'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Google API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Google API key",
                    key="google_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.google_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "google")
                        if success:
                            st.success(message)
                            api_keys['google'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "xai":
            st.markdown("#### 🔑 xAI API Configuration")
            
            # Display beta status and availability note
            if model_info.get("beta_status", False):
                st.warning("⚠️ This model is currently in beta and requires special access")
            
            if "availability_note" in model_info:
                st.info(model_info["availability_note"])
            
            # Add a link to join the waitlist
            st.markdown("""
            **To get access to the Grok model:**
            1. Visit [xAI's website](https://x.ai/)
            2. Join the waitlist
            3. Once approved, you'll receive instructions to get an API key
            """)
            
            current_api_key = st.session_state.get('xai_api_key', api_keys.get('xai', ''))
            
            if current_api_key:
                st.success("✅ xAI API key is already configured")
                if st.button("Change xAI API Key", key="change_xai_key_sidebar"):
                    st.session_state.xai_api_key = ''
                    api_keys['xai'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "xAI API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your xAI API key",
                    key="xai_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.xai_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "xai")
                        if success:
                            st.success(message)
                            api_keys['xai'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
                            if "waitlist" in message.lower():
                                st.markdown("""
                                **It looks like you need to join the waitlist first:**
                                1. Visit [xAI's website](https://x.ai/)
                                2. Join the waitlist
                                3. Once approved, you'll receive instructions to get an API key
                                """)
        
        else:  # HuggingFace models
            st.markdown("#### 🔑 Hugging Face API Configuration")
            current_api_key = st.session_state.get('huggingface_api_key', api_keys.get('huggingface', ''))
            
            if current_api_key:
                st.success("✅ Hugging Face API key is already configured")
                if st.button("Change Hugging Face API Key", key="change_huggingface_key_sidebar"):
                    st.session_state.huggingface_api_key = ''
                    api_keys['huggingface'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Hugging Face API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Hugging Face API key",
                    key="huggingface_api_key_input"
                )
                
                if api_key != current_api_key:
                    if not api_key.startswith('hf_') and api_key != '':
                        st.error("❌ Invalid API key format. Hugging Face API keys should start with 'hf_'")
                    else:
                        st.session_state.huggingface_api_key = api_key
                        if api_key:
                            success, _, message = test_model_access(api_key)
                            if success:
                                st.success(message)
                                api_keys['huggingface'] = api_key
                                save_api_keys(api_keys)
                            else:
                                st.error(message)

    # Get file path from config
    file_path = config.get("announcements_file")
    
    if not file_path:
        st.error("Announcements file path not configured.")
        return
    
    if not os.path.exists(file_path):
        st.error(f"Announcements data file not found: {file_path}")
        return
    
    # Load data
    df = load_announcements_data(file_path)
    if df is None or df.empty:
        st.error("Failed to load announcements data or the file is empty.")
        return
    
    # Get date column name
    date_col = next((col for col in ['Date', 'Announcement_Date', 'Publication_Date'] 
                    if col in df.columns), None)
    
    # Display summary metrics
    if date_col:
        latest_date = df[date_col].max()
        earliest_date = df[date_col].min()
        date_range = f"{earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total", len(df))
        with col2:
            st.metric("🏢 Companies", len(df['Symbol'].unique()) if 'Symbol' in df.columns else "N/A")
        with col3:
            st.metric("📁 Categories", len(df['Category'].unique()) if 'Category' in df.columns else "N/A")
        with col4:
            st.metric("📅 Date Range", date_range)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📰 Recent", "🏢 Company", "📈 Analysis", "🤖 AI Analysis"])
    
    with tab1:
        st.markdown("### 📊 Announcements Summary")
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display recent announcements count
            if date_col:
                days_back = 7
                cutoff_date = datetime.now() - timedelta(days=days_back)
                recent_count = len(df[df[date_col] >= cutoff_date])
                st.metric(f"Announcements in last {days_back} days", recent_count)
            
            # Display top categories
            if 'Category' in df.columns:
                st.markdown("**Top Announcement Categories**")
                category_counts = df['Category'].value_counts().head(5).reset_index()
                category_counts.columns = ['Category', 'Count']
                
                fig = px.bar(
                    category_counts,
                    x='Count',
                    y='Category',
                    orientation='h',
                    title='Top 5 Categories',
                    labels={'Count': 'Number of Announcements', 'Category': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display top companies
            if 'Symbol' in df.columns:
                st.markdown("**Top Companies by Announcements**")
                company_counts = df['Symbol'].value_counts().head(5).reset_index()
                company_counts.columns = ['Symbol', 'Count']
                
                fig = px.bar(
                    company_counts,
                    x='Count',
                    y='Symbol',
                    orientation='h',
                    title='Top 5 Companies',
                    labels={'Count': 'Number of Announcements', 'Symbol': 'Company'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display announcements over time (last 30 days)
            if date_col:
                st.markdown("**Announcements Over Time (Last 30 Days)**")
                days_back = 30
                cutoff_date = datetime.now() - timedelta(days=days_back)
                recent_df = df[df[date_col] >= cutoff_date].copy()
                
                if not recent_df.empty:
                    recent_df['Date_Only'] = recent_df[date_col].dt.date
                    time_series = recent_df.groupby('Date_Only').size().reset_index()
                    time_series.columns = ['Date', 'Count']
                    
                    fig = px.line(
                        time_series,
                        x='Date',
                        y='Count',
                        title='Announcements Over Time',
                        labels={'Date': 'Date', 'Count': 'Number of Announcements'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No announcements in the last 30 days.")
    
    with tab2:
        st.markdown("### 📰 Recent Announcements")
        
        # Create search and filter section
        with st.expander("🔍 Search & Filters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Date range filter
                if date_col:
                    days_back = st.slider("Show announcements from the last N days:", 1, 90, 7, key="recent_days_slider")
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                else:
                    days_back = None
                    cutoff_date = None
                
                # Category filter
                if 'Category' in df.columns:
                    categories = ['All'] + sorted(df['Category'].unique().tolist())
                    selected_category = st.selectbox("Filter by category:", categories, key="recent_category_filter")
                else:
                    selected_category = None
            
            with col2:
                # Company filter
                if 'Symbol' in df.columns:
                    companies = ['All'] + sorted(df['Symbol'].unique().tolist())
                    selected_company = st.selectbox("Filter by company:", companies, key="recent_company_filter")
                else:
                    selected_company = None
                
                # Text search
                search_term = st.text_input("Search in announcements:", key="recent_search_input")
        
        # Apply filters
        filtered_df = df.copy()
        
        if date_col and cutoff_date:
            filtered_df = filtered_df[filtered_df[date_col] >= cutoff_date]
        
        if selected_category and selected_category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        
        if selected_company and selected_company != 'All':
            filtered_df = filtered_df[filtered_df['Symbol'] == selected_company]
        
        if search_term:
            filtered_df = filter_announcements(filtered_df, search_term=search_term)
        
        # Sort by date
        if date_col:
            filtered_df = filtered_df.sort_values(by=date_col, ascending=False)
        
        # Display results
        if not filtered_df.empty:
            st.markdown(f"**Found {len(filtered_df)} announcements**")
            
            # Create a more user-friendly display
            for idx, row in filtered_df.iterrows():
                with st.expander(f"{row['Symbol'] if 'Symbol' in row.index else 'Unknown'} - {row[date_col].strftime('%Y-%m-%d') if date_col else 'No date'}", expanded=False):
                    display_announcement_details(row, date_col)
        else:
            st.info("No announcements found with the selected filters.")
    
    with tab3:
        st.markdown("### 🏢 Company Specific Announcements")
        
        # Company selection
        if 'Symbol' in df.columns:
            companies = sorted(df['Symbol'].unique().tolist())
            selected_company = st.selectbox("Select company:", companies, key="company_specific_company")
            
            # Date range selection
            if date_col:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From date:", value=earliest_date, key="company_specific_start_date")
                with col2:
                    end_date = st.date_input("To date:", value=latest_date, key="company_specific_end_date")
            else:
                start_date = None
                end_date = None
            
            # Category filter
            if 'Category' in df.columns:
                categories = ['All'] + sorted(df['Category'].unique().tolist())
                selected_category = st.selectbox("Filter by category:", categories, key="company_specific_category")
            else:
                selected_category = None
            
            # Apply filters
            company_df = filter_announcements(
                df, 
                company=selected_company,
                category=selected_category if selected_category != 'All' else None,
                start_date=start_date,
                end_date=end_date
            )
            
            # Display company summary
            if not company_df.empty:
                st.markdown(f"**{selected_company} - Announcement Summary**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Announcements", len(company_df))
                with col2:
                    if date_col:
                        latest = company_df[date_col].max()
                        st.metric("Latest Announcement", latest.strftime('%Y-%m-%d'))
                with col3:
                    if 'Category' in company_df.columns:
                        categories = company_df['Category'].unique()
                        st.metric("Categories", len(categories))
                
                # Display announcements
                st.markdown("**Announcements**")
                for idx, row in company_df.iterrows():
                    with st.expander(f"{row[date_col].strftime('%Y-%m-%d') if date_col else 'No date'} - {row['Category'] if 'Category' in row.index else 'No category'}", expanded=False):
                        display_announcement_details(row, date_col)
            else:
                st.info(f"No announcements found for {selected_company} with the selected filters.")
        else:
            st.error("Company data not available in the announcements file.")
    
    with tab4:
        st.markdown("### 📈 Announcements Analysis")
        
        if not df.empty:
            # Create tabs for different analyses
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Categories", "Companies", "Time Series"])
            
            with analysis_tab1:
                # Analyze announcements by category
                if 'Category' in df.columns:
                    st.markdown("**Announcements by Category**")
                    
                    # Allow user to select number of categories to display
                    top_n = st.slider("Number of categories to display:", 5, 30, 10, key="category_analysis_top_n")
                    
                    category_counts = df['Category'].value_counts().nlargest(top_n).reset_index()
                    category_counts.columns = ['Category', 'Count']
                    
                    # Create a pie chart
                    fig_pie = px.pie(
                        category_counts, 
                        values='Count', 
                        names='Category',
                        title=f'Top {top_n} Announcement Categories'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Create a bar chart
                    fig_bar = px.bar(
                        category_counts,
                        x='Category',
                        y='Count',
                        title=f'Top {top_n} Announcement Categories',
                        labels={'Category': 'Category', 'Count': 'Number of Announcements'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Category data not available for analysis.")
            
            with analysis_tab2:
                # Analyze announcements by company
                if 'Symbol' in df.columns:
                    st.markdown("**Announcements by Company**")
                    
                    # Allow user to select number of companies to display
                    top_n = st.slider("Number of companies to display:", 5, 30, 10, key="company_analysis_top_n")
                    
                    company_counts = df['Symbol'].value_counts().nlargest(top_n).reset_index()
                    company_counts.columns = ['Symbol', 'Count']
                    
                    # Create a bar chart
                    fig = px.bar(
                        company_counts,
                        x='Symbol',
                        y='Count',
                        title=f'Top {top_n} Companies by Number of Announcements',
                        labels={'Symbol': 'Company Symbol', 'Count': 'Number of Announcements'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table of top companies
                    st.markdown("**Top Companies Table**")
                    st.dataframe(company_counts, use_container_width=True)
                else:
                    st.info("Company data not available for analysis.")
            
            with analysis_tab3:
                # Analyze announcements over time
                if date_col:
                    st.markdown("**Announcements Over Time**")
                    
                    # Allow user to select time period
                    time_period = st.selectbox(
                        "Select time period:",
                        ["Daily", "Weekly", "Monthly", "Yearly"],
                        key="time_series_period"
                    )
                    
                    # Group by date and count
                    df['Date_Only'] = df[date_col].dt.date
                    
                    if time_period == "Daily":
                        time_series = df.groupby('Date_Only').size().reset_index(name='Count')
                        time_series.columns = ['Date', 'Count']
                    elif time_period == "Weekly":
                        df['Week'] = df[date_col].dt.isocalendar().week
                        df['Year'] = df[date_col].dt.isocalendar().year
                        time_series = df.groupby(['Year', 'Week']).size().reset_index(name='Count')
                        time_series['Date'] = time_series.apply(
                            lambda x: f"{x['Year']}-W{x['Week']}", axis=1
                        )
                        time_series = time_series[['Date', 'Count']]
                    elif time_period == "Monthly":
                        df['Month'] = df[date_col].dt.to_period('M')
                        time_series = df.groupby('Month').size().reset_index(name='Count')
                        time_series['Date'] = time_series['Month'].astype(str)
                        time_series = time_series[['Date', 'Count']]
                    else:  # Yearly
                        df['Year'] = df[date_col].dt.year
                        time_series = df.groupby('Year').size().reset_index(name='Count')
                        time_series['Date'] = time_series['Year'].astype(str)
                        time_series = time_series[['Date', 'Count']]
                    
                    # Create a line chart
                    fig = px.line(
                        time_series,
                        x='Date',
                        y='Count',
                        title=f'Announcements Over Time ({time_period})',
                        labels={'Date': 'Date', 'Count': 'Number of Announcements'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table of time series data
                    st.markdown("**Time Series Data**")
                    st.dataframe(time_series, use_container_width=True)
                else:
                    st.info("Date data not available for time series analysis.")
        else:
            st.info("No data available for analysis.")

    with tab5:
        st.markdown("### 🤖 AI Financial Report Analysis")
        
        if not st.session_state.get('huggingface_api_key'):
            st.warning("Please configure your Hugging Face API key in the sidebar to use the AI analysis feature.")
            st.markdown("""
            To get a Hugging Face API key:
            1. Go to [Hugging Face](https://huggingface.co/)
            2. Create an account or sign in
            3. Go to your profile settings
            4. Navigate to "Access Tokens"
            5. Create a new token with read access
            6. Copy the token and paste it in the sidebar
            """)
            return
        
        # Get enhanced company options with signals
        company_options, default_companies = enhance_company_selection(df)
        
        # Display buy signals summary
        buy_signals = get_latest_buy_signals()
        if buy_signals:
            st.markdown("#### 📈 Latest Buy Signals")
            signals_summary = ""
            for symbol, signal in list(buy_signals.items())[:5]:  # Show top 5 signals
                strength = "🔥" if signal['signal_type'] == 'Strong Buy' else "✨"
                confidence = "⭐" * ({"High": 3, "Medium": 2, "Low": 1}.get(signal['confidence'], 2))
                price_target = f" (Target: {signal['price_target']})" if signal['price_target'] else ""
                signals_summary += f"- {strength} **{symbol}**: {signal['signal_type']}{price_target} {confidence}\n"
            st.markdown(signals_summary)
        
        st.markdown("#### 📊 Select Companies for Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            default_idx1 = next(
                (i for i, opt in enumerate(company_options) 
                 if opt["value"] == default_companies[0]
                ), 0) if default_companies else 0
            
            company1 = st.selectbox(
                "First Company:",
                options=[opt["value"] for opt in company_options],
                format_func=lambda x: next(
                    opt["label"] for opt in company_options if opt["value"] == x
                ),
                index=default_idx1,
                key="ai_company1"
            )
        
        with col2:
            # Allow selecting the same company for second report
            default_idx2 = next(
                (i for i, opt in enumerate(company_options) 
                 if len(default_companies) > 1 and opt["value"] == default_companies[1]
                ), None)
            
            company2 = st.selectbox(
                "Second Company (Optional):",
                options=["None"] + [opt["value"] for opt in company_options],
                format_func=lambda x: "None" if x == "None" else next(
                    opt["label"] for opt in company_options if opt["value"] == x
                ),
                index=1 if default_idx2 is not None else 0,
                key="ai_company2"
            )
        
        # Add information about comparing same company
        if company1 == company2:
            st.info("📌 You've selected the same company for both reports. This allows you to compare different reports from the same company.")
        
        # Auto-select reports for company1
        company1_reports, report1_idx = auto_select_reports(df, company1)
        if not company1_reports.empty:
            st.markdown(f"#### 📄 Reports for {company1}")
            
            report1_options = company1_reports.apply(
                lambda x: f"{x[date_col].strftime('%Y-%m-%d')} - {x['Subject'] if 'Subject' in x.index and pd.notna(x['Subject']) else 'No Subject'}",
                axis=1
            ).tolist()
            
            # Convert indices to regular Python integers for Streamlit
            report_indices = list(range(len(report1_options)))
            
            # Ensure report1_idx is valid
            report1_idx = max(0, min(report1_idx, len(report1_options) - 1))
            
            selected_report1_idx = st.selectbox(
                f"Select report (auto-selected most recent):",
                options=report_indices,
                format_func=lambda x: "📌 " + report1_options[x] if x == report1_idx else report1_options[x],
                index=report1_idx,
                key="ai_report1"
            )
            
            # Show report preview with full Subject
            with st.expander("📄 Preview Selected Report", expanded=False):
                display_announcement_details(company1_reports.iloc[selected_report1_idx], date_col)
            
            # Auto-select reports for company2 if selected
            if company2 != "None":
                company2_reports, report2_idx = auto_select_reports(df, company2)
                if not company2_reports.empty:
                    st.markdown(f"#### 📄 Reports for {company2}")
                    
                    # If same company is selected, exclude the first selected report
                    if company1 == company2:
                        company2_reports = company2_reports.drop(company2_reports.index[selected_report1_idx])
                        if not company2_reports.empty:
                            report2_options = company2_reports.apply(
                                lambda x: f"{x[date_col].strftime('%Y-%m-%d')} - {x['Subject'] if 'Subject' in x.index and pd.notna(x['Subject']) else 'No Subject'}",
                                axis=1
                            ).tolist()
                            
                            # Convert indices to regular Python integers for Streamlit
                            report2_indices = list(range(len(report2_options)))
                            
                            # Ensure report2_idx is valid
                            report2_idx = max(0, min(report2_idx, len(report2_options) - 1))
                            
                            selected_report2_idx = st.selectbox(
                                f"Select report (auto-selected most recent):",
                                options=report2_indices,
                                format_func=lambda x: "📌 " + report2_options[x] if x == report2_idx else report2_options[x],
                                index=report2_idx,
                                key="ai_report2"
                            )
                            
                            # Show report preview with full Subject
                            with st.expander("📄 Preview Selected Report", expanded=False):
                                display_announcement_details(company2_reports.iloc[selected_report2_idx], date_col)
                        else:
                            st.warning("No additional reports available for comparison.")
                            selected_report2_idx = None
                    else:
                        # Regular flow for different companies
                        report2_options = company2_reports.apply(
                            lambda x: f"{x[date_col].strftime('%Y-%m-%d')} - {x['Subject'] if 'Subject' in x.index and pd.notna(x['Subject']) else 'No Subject'}",
                            axis=1
                        ).tolist()
                        
                        # Convert indices to regular Python integers for Streamlit
                        report2_indices = list(range(len(report2_options)))
                        
                        # Ensure report2_idx is valid
                        report2_idx = max(0, min(report2_idx, len(report2_options) - 1))
                        
                        selected_report2_idx = st.selectbox(
                            f"Select report (auto-selected most recent):",
                            options=report2_indices,
                            format_func=lambda x: "📌 " + report2_options[x] if x == report2_idx else report2_options[x],
                            index=report2_idx,
                            key="ai_report2"
                        )
                        
                        # Show report preview with full Subject
                        with st.expander("📄 Preview Selected Report", expanded=False):
                            display_announcement_details(company2_reports.iloc[selected_report2_idx], date_col)
                else:
                    st.info(f"No reports found for {company2}")
                    selected_report2_idx = None
            else:
                selected_report2_idx = None
            
            # Analysis button with loading state
            if st.button("🔍 Analyze Reports", key="analyze_button", use_container_width=True):
                analysis_placeholder = st.empty()
                with st.spinner("🤖 AI is analyzing the reports... This may take a few moments."):
                    try:
                        # Get company 1 info
                        company1_row = company1_reports.iloc[selected_report1_idx]
                        company1_info = {
                            'symbol': company1,
                            'date': company1_row[date_col].strftime('%Y-%m-%d'),
                            'category': company1_row['Category'],
                            'report_type': company1_row.get('Report_Type', 'Financial Report'),
                            'signal': buy_signals.get(company1, {})
                        }
                        
                        # Get report contents
                        report1_content = get_report_content(company1_row)
                        
                        # Get company 2 info if selected
                        company2_info = None
                        report2_content = None
                        if selected_report2_idx is not None:
                            company2_row = company2_reports.iloc[selected_report2_idx]
                            company2_info = {
                                'symbol': company2,
                                'date': company2_row[date_col].strftime('%Y-%m-%d'),
                                'category': company2_row['Category'],
                                'report_type': company2_row.get('Report_Type', 'Financial Report'),
                                'signal': buy_signals.get(company2, {})
                            }
                            report2_content = get_report_content(company2_row)
                        
                        # Get AI analysis
                        analysis = analyze_financial_reports(
                            report1_content,
                            report2_content,
                            company1_info=company1_info,
                            company2_info=company2_info
                        )
                        
                        # Display analysis results
                        with analysis_placeholder.container():
                            st.markdown(analysis)
                            
                            # Add download button for the analysis
                            analysis_filename = f"financial_analysis_{company1}"
                            if company2 != "None":
                                analysis_filename += f"_vs_{company2}"
                            analysis_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            
                            st.download_button(
                                label="📥 Download Analysis",
                                data=analysis.encode(),
                                file_name=analysis_filename,
                                mime="text/markdown",
                                key="download_analysis"
                            )
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.error("Please try again or select different reports.")
        else:
            st.info(f"No reports found for {company1}")

    with st.expander("🔧 Advanced Analysis Settings"):
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        
        # Primary Model Selection with descriptions
        st.markdown("#### Primary Analysis Model")
        primary_model = st.selectbox(
            "Select Primary Model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: f"{x} - {MODEL_OPTIONS[x]['description']}",
            help="Main model for comprehensive analysis"
        )
        
        # Display model strengths
        st.markdown("**Model Strengths:**")
        for strength in MODEL_OPTIONS[primary_model]['strengths']:
            st.markdown(f"- {strength}")
        
        # API Key configuration based on selected model
        model_info = MODEL_OPTIONS[primary_model]
        api_keys = load_api_keys()
        
        if model_info['api_type'] == "deepseek":
            st.markdown("#### 🔑 DeepSeek API Configuration")
            current_api_key = st.session_state.get('deepseek_api_key', api_keys.get('deepseek', ''))
            
            if current_api_key:
                st.success("✅ DeepSeek API key is already configured")
                if st.button("Change DeepSeek API Key", key="change_deepseek_key_advanced"):
                    st.session_state.deepseek_api_key = ''
                    api_keys['deepseek'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "DeepSeek API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your DeepSeek API key",
                    key="deepseek_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.deepseek_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "deepseek")
                        if success:
                            st.success(message)
                            api_keys['deepseek'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "anthropic":
            st.markdown("#### 🔑 Anthropic API Configuration")
            current_api_key = st.session_state.get('anthropic_api_key', api_keys.get('anthropic', ''))
            
            if current_api_key:
                st.success("✅ Anthropic API key is already configured")
                if st.button("Change Anthropic API Key", key="change_anthropic_key_advanced"):
                    st.session_state.anthropic_api_key = ''
                    api_keys['anthropic'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Anthropic API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Anthropic API key",
                    key="anthropic_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.anthropic_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "anthropic")
                        if success:
                            st.success(message)
                            api_keys['anthropic'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "openai":
            st.markdown("#### 🔑 OpenAI API Configuration")
            current_api_key = st.session_state.get('openai_api_key', api_keys.get('openai', ''))
            
            if current_api_key:
                st.success("✅ OpenAI API key is already configured")
                if st.button("Change OpenAI API Key", key="change_openai_key_advanced"):
                    st.session_state.openai_api_key = ''
                    api_keys['openai'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "OpenAI API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your OpenAI API key",
                    key="openai_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.openai_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "openai")
                        if success:
                            st.success(message)
                            api_keys['openai'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "google":
            st.markdown("#### 🔑 Google API Configuration")
            current_api_key = st.session_state.get('google_api_key', api_keys.get('google', ''))
            
            if current_api_key:
                st.success("✅ Google API key is already configured")
                if st.button("Change Google API Key", key="change_google_key_advanced"):
                    st.session_state.google_api_key = ''
                    api_keys['google'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Google API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Google API key",
                    key="google_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.google_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "google")
                        if success:
                            st.success(message)
                            api_keys['google'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
        
        elif model_info['api_type'] == "xai":
            st.markdown("#### 🔑 xAI API Configuration")
            
            # Display beta status and availability note
            if model_info.get("beta_status", False):
                st.warning("⚠️ This model is currently in beta and requires special access")
            
            if "availability_note" in model_info:
                st.info(model_info["availability_note"])
            
            # Add a link to join the waitlist
            st.markdown("""
            **To get access to the Grok model:**
            1. Visit [xAI's website](https://x.ai/)
            2. Join the waitlist
            3. Once approved, you'll receive instructions to get an API key
            """)
            
            current_api_key = st.session_state.get('xai_api_key', api_keys.get('xai', ''))
            
            if current_api_key:
                st.success("✅ xAI API key is already configured")
                if st.button("Change xAI API Key", key="change_xai_key_advanced"):
                    st.session_state.xai_api_key = ''
                    api_keys['xai'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "xAI API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your xAI API key",
                    key="xai_api_key_input"
                )
                
                if api_key != current_api_key:
                    st.session_state.xai_api_key = api_key
                    if api_key:
                        success, _, message = test_model_access(api_key, "xai")
                        if success:
                            st.success(message)
                            api_keys['xai'] = api_key
                            save_api_keys(api_keys)
                        else:
                            st.error(message)
                            if "waitlist" in message.lower():
                                st.markdown("""
                                **It looks like you need to join the waitlist first:**
                                1. Visit [xAI's website](https://x.ai/)
                                2. Join the waitlist
                                3. Once approved, you'll receive instructions to get an API key
                                """)
        
        else:  # HuggingFace models
            st.markdown("#### 🔑 Hugging Face API Configuration")
            current_api_key = st.session_state.get('huggingface_api_key', api_keys.get('huggingface', ''))
            
            if current_api_key:
                st.success("✅ Hugging Face API key is already configured")
                if st.button("Change Hugging Face API Key", key="change_huggingface_key_advanced"):
                    st.session_state.huggingface_api_key = ''
                    api_keys['huggingface'] = ''
                    save_api_keys(api_keys)
                    st.rerun()
            else:
                api_key = st.text_input(
                    "Hugging Face API Key",
                    value=current_api_key,
                    type="password",
                    help="Enter your Hugging Face API key",
                    key="huggingface_api_key_input"
                )
                
                if api_key != current_api_key:
                    if not api_key.startswith('hf_') and api_key != '':
                        st.error("❌ Invalid API key format. Hugging Face API keys should start with 'hf_'")
                    else:
                        st.session_state.huggingface_api_key = api_key
                        if api_key:
                            success, _, message = test_model_access(api_key)
                            if success:
                                st.success(message)
                                api_keys['huggingface'] = api_key
                                save_api_keys(api_keys)
                            else:
                                st.error(message)

        # Analysis Depth
        st.markdown("#### Analysis Configuration")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Detailed"],
            value="Standard",
            help="Controls the level of detail in the analysis"
        )
        
        # Custom Prompting
        enable_custom_prompt = st.checkbox(
            "Enable Custom Analysis Focus",
            help="Customize what aspects of the reports to focus on"
        )
        
        if enable_custom_prompt:
            analysis_focus = st.multiselect(
                "Focus Areas",
                ["Financial Metrics", "Market Position", "Risk Analysis",
                 "Growth Potential", "Competitive Analysis", "Technical Indicators",
                 "Valuation Metrics", "Industry Trends", "Management Quality"],
                default=["Financial Metrics", "Risk Analysis"],
                help="Select specific areas to focus the analysis on"
            )

        # Output Format
        st.markdown("#### Output Configuration")
        output_format = st.radio(
            "Output Format",
            ["Concise", "Detailed", "Technical"],
            horizontal=True,
            help="Choose the style of the analysis output"
        )
        
        # Additional Settings
        st.markdown("#### Additional Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Analysis Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values make the analysis more creative but potentially less focused"
            )
        
        with col2:
            max_tokens = st.slider(
                "Maximum Response Length",
                min_value=500,
                max_value=4000,
                value=MODEL_OPTIONS[primary_model]["max_tokens"],
                step=100,
                help="Maximum length of the analysis response"
            )
        
        # Save settings to session state
        st.session_state.analysis_settings = {
            "primary_model": primary_model,
            "use_secondary": False,
            "secondary_model": None,
            "analysis_depth": analysis_depth,
            "custom_focus": analysis_focus if enable_custom_prompt else None,
            "output_format": output_format,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

def test_model_access(api_key: str, model_type: str = "huggingface") -> tuple[bool, str, str]:
    """
    Test access to available models and return the first accessible one.
    """
    # Silently check version lock without displaying messages
    check_version_lock()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if model_type == "deepseek":
        try:
            test_payload = {
                "model": MODEL_OPTIONS["DeepSeek"]["model_name"],
                "messages": [{"role": "user", "content": "Test model access"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                MODEL_OPTIONS["DeepSeek"]["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, MODEL_OPTIONS["DeepSeek"]["url"], "✅ Successfully connected to DeepSeek model"
            elif response.status_code == 401:
                return False, "", "❌ Invalid DeepSeek API key. Please check your key and ensure it's correct."
            else:
                return False, "", f"❌ Error accessing DeepSeek API: {response.text}"
                
        except Exception as e:
            return False, "", f"❌ Error testing DeepSeek model: {str(e)}"
    
    elif model_type == "anthropic":
        try:
            headers["anthropic-version"] = "2023-06-01"
            test_payload = {
                "model": MODEL_OPTIONS["Claude-3"]["model_name"],
                "messages": [{"role": "user", "content": "Test model access"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                MODEL_OPTIONS["Claude-3"]["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, MODEL_OPTIONS["Claude-3"]["url"], "✅ Successfully connected to Claude model"
            elif response.status_code == 401:
                return False, "", "❌ Invalid Anthropic API key. Please check your key and ensure it's correct."
            else:
                return False, "", f"❌ Error accessing Anthropic API: {response.text}"
                
        except Exception as e:
            return False, "", f"❌ Error testing Claude model: {str(e)}"
    
    elif model_type == "openai":
        try:
            test_payload = {
                "model": MODEL_OPTIONS["GPT-4"]["model_name"],
                "messages": [{"role": "user", "content": "Test model access"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                MODEL_OPTIONS["GPT-4"]["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, MODEL_OPTIONS["GPT-4"]["url"], "✅ Successfully connected to GPT-4 model"
            elif response.status_code == 401:
                return False, "", "❌ Invalid OpenAI API key. Please check your key and ensure it's correct."
            else:
                return False, "", f"❌ Error accessing OpenAI API: {response.text}"
                
        except Exception as e:
            return False, "", f"❌ Error testing GPT-4 model: {str(e)}"
    
    elif model_type == "google":
        try:
            test_payload = {
                "contents": [{"parts": [{"text": "Test model access"}]}]
            }
            
            response = requests.post(
                MODEL_OPTIONS["Gemini-Pro"]["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, MODEL_OPTIONS["Gemini-Pro"]["url"], "✅ Successfully connected to Gemini model"
            elif response.status_code == 401:
                return False, "", "❌ Invalid Google API key. Please check your key and ensure it's correct."
            else:
                return False, "", f"❌ Error accessing Google API: {response.text}"
                
        except Exception as e:
            return False, "", f"❌ Error testing Gemini model: {str(e)}"
    
    elif model_type == "xai":
        try:
            test_payload = {
                "model": MODEL_OPTIONS["Grok-1"]["model_name"],
                "messages": [{"role": "user", "content": "Test model access"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                MODEL_OPTIONS["Grok-1"]["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, MODEL_OPTIONS["Grok-1"]["url"], "✅ Successfully connected to Grok model"
            elif response.status_code == 401:
                return False, "", "❌ Invalid xAI API key. Please check your key and ensure it's correct."
            elif response.status_code == 404:
                try:
                    error_data = response.json()
                    if "error" in error_data and "model" in error_data["error"]:
                        return False, "", """❌ The Grok model is not available with your current API key.
                        
This could be due to:
1. Your account doesn't have access to the Grok model yet
2. You need to join the xAI waitlist at https://x.ai/
3. Your API key is for a different xAI service

Please visit https://x.ai/ to join the waitlist or check your API key."""
                    else:
                        return False, "", f"❌ Error accessing xAI API: {response.text}"
                except:
                    return False, "", f"❌ Error accessing xAI API: {response.text}"
            else:
                return False, "", f"❌ Error accessing xAI API: {response.text}"
            
        except Exception as e:
            return False, "", f"❌ Error testing Grok model: {str(e)}"
    
    # Default HuggingFace models test
    for model_name, model_info in MODEL_OPTIONS.items():
        if model_info["api_type"] != "huggingface":
            continue
            
        try:
            test_payload = {
                "inputs": "Test model access",
                "parameters": {"max_new_tokens": 5}
            }
            
            response = requests.post(
                model_info["url"],
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, model_info["url"], f"✅ Successfully connected to {model_name} model"
            elif response.status_code == 403:
                continue
            elif response.status_code == 401:
                return False, "", "❌ Invalid API key. Please check your key and ensure it's correct."
            
        except Exception:
            continue
    
    return False, "", """❌ Could not access any models. Please ensure:
1. You're logged into the respective platform
2. You've accepted the model terms
3. Your API key has proper access
4. You've waited a few minutes after accepting terms"""

def analyze_financial_reports(report1_text: str, report2_text: str = None, 
                        company1_info: Dict = None, company2_info: Dict = None) -> str:
    """
    Use AI models to analyze financial reports and provide investment advice.
    """
    # Silently check version lock without displaying messages
    check_version_lock()
    
    # Get API keys from session state
    huggingface_api_key = st.session_state.get('huggingface_api_key', '')
    deepseek_api_key = st.session_state.get('deepseek_api_key', '')
    
    # Get selected model from session state
    selected_model = st.session_state.get('selected_model', 'Mistral-7B')
    model_info = MODEL_OPTIONS[selected_model]
    
    if model_info["api_type"] == "deepseek" and not deepseek_api_key:
        return "⚠️ Error: DeepSeek API key not configured. Please set up your DeepSeek API key in the sidebar."
    
    if model_info["api_type"] == "huggingface" and not huggingface_api_key:
        return "⚠️ Error: Hugging Face API key not configured. Please set up your Hugging Face API key in the sidebar."
    
    try:
        # Get full company names
        company1_name = company1_info.get('company_name', company1_info['symbol'])
        company2_name = company2_info.get('company_name', company2_info['symbol']) if company2_info else None
        
        # Prepare the prompt
        if report2_text and company2_info:
            prompt = f"""You are a professional financial analyst. Analyze these two financial reports:

Company 1: {company1_name} ({company1_info['symbol']})
Report Date: {company1_info['date']}
Report Type: {company1_info['category']}
Report Content:
{report1_text[:3000]}

Company 2: {company2_name} ({company2_info['symbol']})
Report Date: {company2_info['date']}
Report Type: {company2_info['category']}
Report Content:
{report2_text[:3000]}

Provide a detailed comparative analysis in the following format:
1. SUMMARY
- {company1_name} ({company1_info['symbol']}): Brief summary of key points (2-3 sentences)
- {company2_name} ({company2_info['symbol']}): Brief summary of key points (2-3 sentences)

2. KEY METRICS COMPARISON
- Revenue and Growth comparison
- Profitability metrics comparison
- Debt and Liquidity comparison
- Key differences between the companies

3. FUNDAMENTAL ANALYSIS
- {company1_name} ({company1_info['symbol']}) strengths and weaknesses
- {company2_name} ({company2_info['symbol']}) strengths and weaknesses
- Competitive position in the market
- Industry outlook

4. INVESTMENT RECOMMENDATION
- {company1_name} ({company1_info['symbol']}): Recommendation (Buy/Hold/Sell) with rationale
- {company2_name} ({company2_info['symbol']}): Recommendation (Buy/Hold/Sell) with rationale
- Risk factors for each company
- Comparative investment perspective
"""
        else:
            prompt = f"""You are a professional financial analyst. Analyze this financial report:

Company: {company1_name} ({company1_info['symbol']})
Report Date: {company1_info['date']}
Report Type: {company1_info['category']}
Report Content:
{report1_text[:6000]}

Provide a detailed analysis in the following format:
1. SUMMARY
- Brief summary of {company1_name} ({company1_info['symbol']})'s report (2-3 sentences)
- Key highlights and important announcements

2. KEY METRICS
- Revenue and Growth analysis
- Profitability metrics
- Balance sheet highlights
- Cash flow analysis

3. FUNDAMENTAL ANALYSIS
- {company1_name} ({company1_info['symbol']})'s strengths
- {company1_name} ({company1_info['symbol']})'s challenges
- Market position and competitive advantages
- Future outlook and growth potential

4. INVESTMENT RECOMMENDATION
- Recommendation for {company1_name} ({company1_info['symbol']}) (Buy/Hold/Sell)
- Detailed rationale for the recommendation
- Risk factors to consider
- Price targets and timeline (if applicable)
"""
        
        if model_info["api_type"] == "deepseek":
            # Call DeepSeek API directly
            headers = {
                "Authorization": f"Bearer {deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_info["model_name"],
                "messages": [
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": model_info["max_tokens"],
                "temperature": 0.7
            }
            
            response = requests.post(
                model_info["url"],
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    analysis = response.json()['choices'][0]['message']['content']
                except Exception as e:
                    return f"❌ Error parsing DeepSeek response: {str(e)}\nResponse: {response.text}"
            else:
                error_msg = f"⚠️ DeepSeek API request failed with status code {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f"\nDetails: {error_details}"
                except:
                    error_msg += f"\nResponse: {response.text}"
                return error_msg
        
        else:  # HuggingFace models
            headers = {
                "Authorization": f"Bearer {huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": model_info["max_tokens"],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                model_info["url"],
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    if isinstance(response.json(), list):
                        analysis = response.json()[0].get('generated_text', '')
                    else:
                        analysis = response.json().get('generated_text', '')
                except Exception as e:
                    return f"❌ Error parsing model response: {str(e)}\nResponse: {response.text}"
            else:
                error_msg = f"⚠️ API request failed with status code {response.status_code}"
                try:
                    error_details = response.json()
                    if response.status_code == 429:
                        return "⌛ The API is currently busy. Please wait a few moments and try again."
                    elif response.status_code == 403:
                        return """❌ Access denied. Please ensure you have:
1. Accepted the model's terms of use
2. Waited a few minutes after accepting terms
3. Have proper API access permissions"""
                    error_msg += f"\nDetails: {error_details}"
                except:
                    error_msg += f"\nResponse: {response.text}"
                return error_msg

        # Format the analysis
        formatted_analysis = format_analysis_output(analysis, company1_info, company2_info)
        return formatted_analysis

    except requests.exceptions.Timeout:
        return "⏱️ Error: API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"🌐 Error making API request: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error during analysis: {str(e)}"

def format_analysis_output(analysis: str, company1_info: Dict, company2_info: Dict = None) -> str:
    """Format the analysis output with markdown styling."""
    # Silently check version lock without displaying messages
    check_version_lock()
    
    # Get full company names
    company1_name = company1_info.get('company_name', company1_info['symbol'])
    company2_name = company2_info.get('company_name', company2_info['symbol']) if company2_info else None
    
    formatted_analysis = f"## 📊 Financial Report Analysis"
    if company2_info:
        formatted_analysis += f" - {company1_name} ({company1_info['symbol']}) vs {company2_name} ({company2_info['symbol']})"
    else:
        formatted_analysis += f" - {company1_name} ({company1_info['symbol']})"
    formatted_analysis += "\n\n"
    
    # Split analysis into sections and format
    sections = analysis.split('\n\n')
    for section in sections:
        if section.startswith('1. SUMMARY'):
            formatted_analysis += "### 📝 Summary\n" + section.replace('1. SUMMARY', '') + "\n\n"
        elif section.startswith('2. KEY METRICS'):
            formatted_analysis += "### 📈 Key Metrics\n" + section.replace('2. KEY METRICS', '').replace('2. KEY METRICS COMPARISON', '') + "\n\n"
        elif section.startswith('3. FUNDAMENTAL'):
            formatted_analysis += "### 🔍 Fundamental Analysis\n" + section.replace('3. FUNDAMENTAL ANALYSIS', '') + "\n\n"
        elif section.startswith('4. INVESTMENT'):
            formatted_analysis += "### 💡 Investment Recommendation\n" + section.replace('4. INVESTMENT RECOMMENDATION', '') + "\n\n"
        else:
            formatted_analysis += section + "\n\n"
    
    # Add analysis metadata
    formatted_analysis += "\n---\n"
    formatted_analysis += f"*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    formatted_analysis += f"*Report Types: {company1_info['category']}"
    if company2_info:
        formatted_analysis += f" vs {company2_info['category']}"
    formatted_analysis += "*"
    
    return formatted_analysis

def get_report_content(row: pd.Series) -> str:
    """Extract text content from a report."""
    # Silently check version lock without displaying messages
    check_version_lock()
    
    try:
        # First try to get content from report URL
        report_url = get_report_url(row)
        if report_url:
            content = download_file(report_url)
            if content:
                # Extract text based on file type
                if is_pdf_file(report_url):
                    return extract_text_from_pdf(content)
                elif is_docx_file(report_url):
                    return extract_text_from_docx(content)
                else:
                    return content.decode('utf-8', errors='ignore')
        
        # If no report URL or couldn't extract text, use announcement content
        text = ""
        content_columns = ['Details', 'Announcement', 'Description', 'Content']
        for col in content_columns:
            if col in row.index and pd.notna(row[col]):
                text += f"{row[col]}\n\n"
        
        return text
    
    except Exception as e:
        return f"Error extracting report content: {str(e)}"

MODEL_OPTIONS = {
    "Mistral-7B": {
        "url": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Best for detailed financial analysis, good balance of speed and accuracy",
        "max_tokens": 1500,
        "strengths": ["Detailed analysis", "Good financial understanding", "Fast responses"],
        "api_type": "huggingface"
    },
    "DeepSeek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "description": "Advanced model for in-depth financial analysis and technical insights",
        "max_tokens": 2000,
        "model_name": "deepseek-chat",
        "strengths": ["Deep financial expertise", "Technical analysis", "Complex reasoning"],
        "api_type": "deepseek"
    },
    "Llama-2-70B": {
        "url": "meta-llama/Llama-2-70b-chat-hf",
        "description": "Most comprehensive analysis, best for complex reports",
        "max_tokens": 2000,
        "strengths": ["Deep financial knowledge", "Complex reasoning", "Thorough analysis"],
        "api_type": "huggingface"
    },
    "MPT-30B": {
        "url": "mosaicml/mpt-30b-instruct",
        "description": "Specialized in financial metrics and ratios",
        "max_tokens": 1500,
        "strengths": ["Financial metrics focus", "Good with numbers", "Technical analysis"],
        "api_type": "huggingface"
    },
    "FLAN-T5": {
        "url": "google/flan-t5-xl",
        "description": "Fast analysis, good for quick insights",
        "max_tokens": 1000,
        "strengths": ["Quick analysis", "Good summarization", "Reliable performance"],
        "api_type": "huggingface"
    },
    "Claude-3": {
        "url": "https://api.anthropic.com/v1/messages",
        "description": "Advanced reasoning and analysis capabilities",
        "max_tokens": 4000,
        "model_name": "claude-3-opus-20240229",
        "strengths": ["Advanced reasoning", "Comprehensive analysis", "Context understanding"],
        "api_type": "anthropic"
    },
    "GPT-4": {
        "url": "https://api.openai.com/v1/chat/completions",
        "description": "State-of-the-art language model with strong analytical capabilities",
        "max_tokens": 4000,
        "model_name": "gpt-4-turbo-preview",
        "strengths": ["Advanced analysis", "Broad knowledge", "Complex reasoning"],
        "api_type": "openai"
    },
    "Gemini-Pro": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "description": "Google's advanced AI model with strong analytical capabilities",
        "max_tokens": 2048,
        "model_name": "gemini-pro",
        "strengths": ["Data analysis", "Technical insights", "Multimodal capabilities"],
        "api_type": "google"
    },
    "Grok-1": {
        "url": "https://api.x.ai/v1/chat/completions",
        "description": "xAI's advanced model with strong analytical and reasoning capabilities (Currently in beta/waitlist)",
        "max_tokens": 4096,
        "model_name": "grok-1",
        "strengths": ["Advanced reasoning", "Real-time analysis", "Complex problem solving"],
        "api_type": "xai",
        "availability_note": "This model is currently in beta and requires joining the xAI waitlist at https://x.ai/",
        "beta_status": True
    }
}

def get_ensemble_analysis(reports, models=None, weights=None):
    """
    Conceptual structure for ensemble analysis using multiple models
    
    analyses = {
        "metrics": {
            "mistral": use_mistral_for_metrics(),
            "llama": use_llama_for_metrics(),
            "mpt": use_mpt_for_metrics()
        },
        "recommendations": {
            "mistral": get_mistral_recommendations(),
            "llama": get_llama_recommendations(),
            "mpt": get_mpt_recommendations()
        }
    }
    
    final_analysis = combine_analyses(analyses, weights)
    """
    pass

ANALYSIS_STYLES = {
    "Standard": {
        "description": "Balanced analysis of all aspects",
        "sections": ["Summary", "Metrics", "Analysis", "Recommendation"]
    },
    "Technical": {
        "description": "Focus on financial metrics and technical analysis",
        "sections": ["Technical Indicators", "Financial Ratios", "Trend Analysis"]
    },
    "Fundamental": {
        "description": "Deep dive into company fundamentals",
        "sections": ["Business Model", "Market Position", "Growth Prospects"]
    },
    "Comparative": {
        "description": "Detailed comparison between companies",
        "sections": ["Metric Comparison", "Competitive Analysis", "Relative Valuation"]
    }
}

def get_latest_buy_signals() -> Dict[str, Dict]:
    """
    Get the latest buy recommendations from investing signals.
    Returns a dictionary of companies with buy signals and their details.
    """
    try:
        # Try to get signals from the signals module
        try:
            from ..signals import get_latest_signals
            signals_df = get_latest_signals()
        except ImportError:
            # Fallback to local signals file if module not available
            signals_file = os.path.join(os.path.dirname(__file__), "..", "data", "investing_signals.xlsx")
            if os.path.exists(signals_file):
                signals_df = pd.read_excel(signals_file)
            else:
                return {}
        
        # Filter for buy signals
        buy_signals = signals_df[
            (signals_df['Signal'].str.contains('Buy', case=False, na=False)) |
            (signals_df['Signal'].str.contains('Strong Buy', case=False, na=False))
        ].copy()
        
        # Sort by date and signal strength
        buy_signals['Signal_Strength'] = buy_signals['Signal'].map({
            'Strong Buy': 2,
            'Buy': 1
        }).fillna(0)
        
        buy_signals = buy_signals.sort_values(
            ['Date', 'Signal_Strength'], 
            ascending=[False, False]
        )
        
        # Convert to dictionary with company symbols as keys
        signals_dict = {}
        for _, row in buy_signals.iterrows():
            signals_dict[row['Symbol']] = {
                'signal_date': row['Date'],
                'signal_strength': row['Signal_Strength'],
                'signal_type': row['Signal'],
                'price_target': row.get('Price_Target', None),
                'confidence': row.get('Confidence', 'Medium')
            }
        
        return signals_dict
    except Exception as e:
        st.warning(f"Could not load latest buy signals: {str(e)}")
        return {}

def get_latest_financial_reports(df: pd.DataFrame, company: str, n: int = 2) -> pd.DataFrame:
    """
    Get the n most recent financial reports for a company.
    """
    # Silently check version lock without displaying messages
    check_version_lock()
    
    company_reports = df[df['Symbol'] == company].copy()
    date_col = next((col for col in ['Date', 'Announcement_Date', 'Publication_Date'] 
                     if col in company_reports.columns), None)
    
    if date_col:
        return company_reports.nlargest(n, date_col)
    return company_reports.head(n)

def enhance_company_selection(df: pd.DataFrame):
    """
    Enhance company selection with buy signals and auto-select latest reports.
    """
    # Silently check version lock without displaying messages
    check_version_lock()
    
    # Get latest buy signals
    buy_signals = get_latest_buy_signals()
    
    # Create formatted company options with signal indicators and full names
    company_options = []
    for symbol in sorted(df['Symbol'].unique()):
        # Get full company name from the dataframe
        company_name = df[df['Symbol'] == symbol]['Company_Name'].iloc[0] if 'Company_Name' in df.columns else symbol
        
        if symbol in buy_signals:
            signal = buy_signals[symbol]
            strength = "🔥" if signal['signal_type'] == 'Strong Buy' else "✨"
            confidence = "⭐" * ({"High": 3, "Medium": 2, "Low": 1}.get(signal['confidence'], 2))
            label = f"{company_name} ({symbol}) {strength} ({signal['signal_type']}) {confidence}"
        else:
            label = f"{company_name} ({symbol})"
        company_options.append({"label": label, "value": symbol})
    
    # Pre-select companies with strongest buy signals
    default_companies = [
        symbol for symbol in buy_signals.keys()
        if buy_signals[symbol]['signal_type'] == 'Strong Buy'
    ][:2]  # Get top 2 companies with strong buy signals
    
    if len(default_companies) < 2:
        # Add companies with regular buy signals if needed
        additional_companies = [
            symbol for symbol in buy_signals.keys()
            if symbol not in default_companies
        ][:2 - len(default_companies)]
        default_companies.extend(additional_companies)
    
    return company_options, default_companies

def auto_select_reports(df: pd.DataFrame, company: str) -> tuple[pd.DataFrame, int]:
    """
    Automatically select the most recent relevant reports for analysis.
    Returns the filtered reports and the index of the recommended report.
    """
    # Silently check version lock without displaying messages
    check_version_lock()
    
    company_reports = df[df['Symbol'] == company].copy()
    
    # Get date column
    date_col = next((col for col in ['Date', 'Announcement_Date', 'Publication_Date'] 
                     if col in company_reports.columns), None)
    
    if not date_col or company_reports.empty:
        return company_reports, 0
    
    # Prioritize certain report types
    priority_types = [
        'Financial Results',
        'Quarterly Report',
        'Annual Report',
        'Financial Statements',
        'Half Yearly Report',
        'Year End Report'
    ]
    
    # Score reports based on recency and type
    company_reports['report_score'] = company_reports.apply(
        lambda x: (
            (pd.Timestamp.now() - x[date_col]).days * -1  # Newer is better
            + (10000 if any(pt.lower() in str(x.get('Category', '')).lower() 
                           for pt in priority_types) else 0)  # Priority boost
            + (5000 if 'quarterly' in str(x.get('Category', '')).lower() else 0)  # Quarterly boost
            + (7000 if 'annual' in str(x.get('Category', '')).lower() else 0)  # Annual boost
        ),
        axis=1
    )
    
    try:
        # Get the highest scored report and convert to regular Python int
        best_report_idx = int(company_reports['report_score'].idxmax())
        # Ensure index is valid
        if best_report_idx < 0 or best_report_idx >= len(company_reports):
            best_report_idx = 0
    except:
        best_report_idx = 0
    
    return company_reports, best_report_idx 
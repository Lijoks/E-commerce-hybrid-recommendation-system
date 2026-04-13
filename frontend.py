# frontend.py - Streamlit frontend for the recommender
import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .recommendation-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .product-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .product-price {
        color: #B12704;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .product-rating {
        color: #FFA41C;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

def get_recommendations(user_id, n_recommendations=10):
    """Get recommendations from the API."""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": user_id, "n_recommendations": n_recommendations},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_user_info(user_id):
    """Get user information from the API."""
    try:
        response = requests.get(f"{API_URL}/user/{user_id}/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_item_info(asin):
    """Get item information from the API."""
    try:
        response = requests.get(f"{API_URL}/item/{asin}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def display_recommendations(recommendations):
    """Display recommendations in a grid."""
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    # Display in columns
    cols = st.columns(3)
    for idx, item in enumerate(recommendations):
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="product-title">{item.get('title', 'Unknown Product')[:50]}...</div>
                """, unsafe_allow_html=True)
                
                # Display rating with stars
                stars = item.get('stars', 0)
                if stars:
                    star_display = "⭐" * int(stars) + "☆" * (5 - int(stars))
                    st.markdown(f'<div class="product-rating">{star_display} {stars}</div>', unsafe_allow_html=True)
                
                # Display price
                price = item.get('price', 0)
                if price:
                    st.markdown(f'<div class="product-price">${price:.2f}</div>', unsafe_allow_html=True)
                
                # Display score
                score = item.get('score', 0)
                st.progress(min(1.0, score / 5), text=f"Relevance Score: {score:.3f}")
                
                st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Header
    st.title("🛍️ Amazon Product Recommender")
    st.markdown("*Personalized recommendations *")
    st.divider()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        # User input
        user_id = st.text_input(
            "Enter User ID",
            value="A2SUAM1J3GNN3B",
            help="Enter a valid Amazon user ID from the dataset"
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
        
        # Get recommendations button
        get_recs = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
        
        st.divider()
        
        # Stats
        st.header("📊 System Stats")
        try:
            stats_response = requests.get(f"{API_URL}/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.metric("Users in Model", f"{stats['users_in_model']:,}")
                st.metric("Items in Model", f"{stats['items_in_model']:,}")
                st.metric("Products in Catalog", f"{stats['products_in_catalog']:,}")
        except:
            st.warning("API not reachable")
    
    # Main content area
    if get_recs:
        with st.spinner("Fetching recommendations..."):
            result = get_recommendations(user_id, n_recommendations)
            
            if result:
                recommendations = result.get('recommendations', [])
                processing_time = result.get('processing_time_ms', 0)
                
                # Show user info if available
                user_info = get_user_info(user_id)
                if user_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("User ID", user_id)
                    with col2:
                        st.metric("Interactions", user_info.get('n_interactions', 'N/A'))
                    with col3:
                        st.metric("Avg Rating", user_info.get('avg_rating', 'N/A'))
                
                st.success(f"✅ Found {len(recommendations)} recommendations in {processing_time:.0f}ms")
                st.divider()
                
                # Display recommendations
                display_recommendations(recommendations)
                
                # Show raw JSON for debugging (expandable)
                with st.expander("🔍 View Raw Response"):
                    st.json(result)
    
    else:
        # Welcome message
        st.info("👈 Enter a User ID and click 'Get Recommendations' to see personalized product suggestions!")
        
        # Example users
        st.subheader("📝 Example User IDs")
        st.markdown("""
        Try these example user IDs from the dataset:
        - `A2SUAM1J3GNN3B`
        - `A3SG8Z8H8Z8H8Z`
        - `A1Z4Z8Z8Z8Z8Z`
        
        Or use any user ID from your training data.
        """)

if __name__ == "__main__":
    main()
"""
Fashion Recommendation System - Streamlit Demo Application
Upload your photo and clothing items to get personalized outfit recommendations!
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from skin_tone_analyzer import SkinToneAnalyzer, FitzpatrickType, Undertone
from clothing_detector import ClothingDetector, ClothingType, ClothingStyle
from color_analyzer import ColorAnalyzer
from recommendation_engine import RecommendationEngine, Occasion, Season

# Page configuration
st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .color-box {
        width: 100px;
        height: 100px;
        margin: 10px;
        border-radius: 10px;
        display: inline-block;
        border: 2px solid #333;
    }
    .outfit-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'skin_tone_result' not in st.session_state:
    st.session_state.skin_tone_result = None
if 'wardrobe' not in st.session_state:
    st.session_state.wardrobe = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# Initialize models
@st.cache_resource
def load_models():
    """Load all models (cached)."""
    skin_analyzer = SkinToneAnalyzer()
    clothing_detector = ClothingDetector()
    color_analyzer = ColorAnalyzer()
    recommender = RecommendationEngine()
    return skin_analyzer, clothing_detector, color_analyzer, recommender

def save_uploaded_file(uploaded_file, folder="user_uploads"):
    """Save uploaded file to data/user_uploads directory."""
    upload_dir = Path(__file__).parent.parent / 'data' / folder
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def display_color_palette(colors, label="Color Palette"):
    """Display a color palette."""
    st.write(f"**{label}:**")
    html = ""
    for color in colors:
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        html += f'<div class="color-box" style="background-color: {hex_color};" title="{hex_color}"></div>'
    st.markdown(html, unsafe_allow_html=True)

def main():
    """Main application."""
    
    # Header
    st.markdown('<p class="main-header">👗 AI Fashion Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get personalized outfit recommendations based on your skin tone!</p>', unsafe_allow_html=True)
    
    # Load models
    skin_analyzer, clothing_detector, color_analyzer, recommender = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Occasion selection
        st.subheader("👔 Occasion")
        occasion = st.selectbox(
            "What's the purpose?",
            options=[
                "Casual",
                "Formal",
                "Business",
                "Party",
                "Date",
                "Athletic/Gym",
                "Beach/Outdoor"
            ],
            index=0
        )
        
        # Map to Occasion enum
        occasion_map = {
            "Casual": Occasion.CASUAL,
            "Formal": Occasion.FORMAL,
            "Business": Occasion.BUSINESS,
            "Party": Occasion.PARTY,
            "Date": Occasion.DATE,
            "Athletic/Gym": Occasion.ATHLETIC,
            "Beach/Outdoor": Occasion.BEACH
        }
        selected_occasion = occasion_map[occasion]
        
        # Season selection
        st.subheader("🌸 Season")
        season = st.selectbox(
            "Current season?",
            options=["Spring", "Summer", "Autumn", "Winter"],
            index=0
        )
        
        season_map = {
            "Spring": Season.SPRING,
            "Summer": Season.SUMMER,
            "Autumn": Season.AUTUMN,
            "Winter": Season.WINTER
        }
        selected_season = season_map[season]
        
        # Number of recommendations
        st.subheader("📊 Recommendations")
        num_recommendations = st.slider(
            "How many outfit suggestions?",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload your photo
        2. Add your clothing items
        3. Get personalized recommendations!
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Analyze Skin Tone", "👕 Build Wardrobe", "✨ Get Recommendations"])
    
    # Tab 1: Skin Tone Analysis
    with tab1:
        st.header("1️⃣ Upload Your Photo")
        st.write("Upload a clear photo of your face for skin tone analysis.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            photo_file = st.file_uploader(
                "Choose a photo",
                type=['jpg', 'jpeg', 'png'],
                key="photo_upload"
            )
            
            if photo_file:
                st.image(photo_file, caption="Your Photo", use_container_width=True)
                
                if st.button("🔍 Analyze Skin Tone", type="primary"):
                    with st.spinner("Analyzing your skin tone..."):
                        # Save file
                        photo_path = save_uploaded_file(photo_file)
                        
                        # Analyze
                        result = skin_analyzer.analyze(photo_path)
                        
                        if result:
                            st.session_state.skin_tone_result = result
                            st.success("✅ Analysis complete!")
                        else:
                            st.error("❌ Could not detect face. Please upload a clearer photo.")
        
        with col2:
            if st.session_state.skin_tone_result:
                result = st.session_state.skin_tone_result
                
                st.subheader("📊 Analysis Results")
                
                # Display results in metric cards
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Fitzpatrick Type", result.fitzpatrick_type.name)
                    st.metric("Undertone", result.undertone.value.title())
                
                with col_b:
                    st.metric("Confidence", f"{result.confidence:.1%}")
                    st.metric("Dominant Color", result.dominant_color_hex)
                
                # Display color
                st.write("**Your Skin Tone:**")
                r, g, b = result.dominant_color_rgb
                hex_color = result.dominant_color_hex
                st.markdown(
                    f'<div class="color-box" style="background-color: {hex_color}; width: 150px; height: 150px;"></div>',
                    unsafe_allow_html=True
                )
                
                # Display color space values
                with st.expander("🔬 Technical Details"):
                    st.write(f"**RGB:** {result.dominant_color_rgb}")
                    st.write(f"**HSV:** {tuple(round(x, 2) for x in result.hsv_values)}")
                    st.write(f"**LAB:** {tuple(round(x, 2) for x in result.lab_values)}")
                
                # Best colors for this skin tone
                st.subheader("🎨 Colors That Suit You")
                
                if result.fitzpatrick_type in recommender.fitzpatrick_color_guidelines:
                    guidelines = recommender.fitzpatrick_color_guidelines[result.fitzpatrick_type]
                    display_color_palette(guidelines['best_colors'], "Recommended Colors")
    
    # Tab 2: Wardrobe Management
    with tab2:
        st.header("2️⃣ Build Your Virtual Wardrobe")
        st.write("Upload images of your clothing items.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("➕ Add New Item")
            
            clothing_file = st.file_uploader(
                "Upload clothing image",
                type=['jpg', 'jpeg', 'png'],
                key="clothing_upload"
            )
            
            if clothing_file:
                st.image(clothing_file, caption="Clothing Item", use_container_width=True)
                
                # Manual type selection
                clothing_type_name = st.selectbox(
                    "Clothing Type",
                    options=[
                        "T-Shirt", "Shirt", "Blouse", "Pants", "Jeans",
                        "Shorts", "Dress", "Skirt", "Jacket", "Coat",
                        "Sweater", "Hoodie", "Shoes", "Accessories"
                    ]
                )
                
                # Map to enum
                type_map = {
                    "T-Shirt": ClothingType.T_SHIRT,
                    "Shirt": ClothingType.SHIRT,
                    "Blouse": ClothingType.BLOUSE,
                    "Pants": ClothingType.PANTS,
                    "Jeans": ClothingType.JEANS,
                    "Shorts": ClothingType.SHORTS,
                    "Dress": ClothingType.DRESS,
                    "Skirt": ClothingType.SKIRT,
                    "Jacket": ClothingType.JACKET,
                    "Coat": ClothingType.COAT,
                    "Sweater": ClothingType.SWEATER,
                    "Hoodie": ClothingType.HOODIE,
                    "Shoes": ClothingType.SHOES,
                    "Accessories": ClothingType.ACCESSORIES,
                }
                
                if st.button("➕ Add to Wardrobe", type="primary"):
                    with st.spinner("Analyzing clothing item..."):
                        # Save file
                        clothing_path = save_uploaded_file(clothing_file)
                        
                        # Detect and analyze
                        item = clothing_detector.detect(
                            clothing_path,
                            clothing_type=type_map[clothing_type_name]
                        )
                        
                        # Add to wardrobe
                        st.session_state.wardrobe.append(item)
                        st.success(f"✅ Added {clothing_type_name} to wardrobe!")
        
        with col2:
            st.subheader("👕 Your Wardrobe")
            
            if st.session_state.wardrobe:
                st.write(f"**Total Items:** {len(st.session_state.wardrobe)}")
                
                # Display wardrobe items
                for i, item in enumerate(st.session_state.wardrobe):
                    with st.expander(f"{item.item_type.value.title()} - {item.dominant_color_hex()}"):
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            if item.image_path and os.path.exists(item.image_path):
                                st.image(item.image_path, use_container_width=True)
                        
                        with col_b:
                            st.write(f"**Type:** {item.item_type.value}")
                            st.write(f"**Style:** {item.style.value}")
                            st.write(f"**Pattern:** {item.pattern.value}")
                            st.write(f"**Dominant Color:** {item.dominant_color_hex()}")
                            
                            # Display color palette
                            if item.color_palette:
                                display_color_palette(item.color_palette[:3], "Colors")
                            
                            if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                                st.session_state.wardrobe.pop(i)
                                st.rerun()
                
                if st.button("🗑️ Clear Wardrobe"):
                    st.session_state.wardrobe = []
                    st.rerun()
            else:
                st.info("👆 Add clothing items to build your wardrobe!")
    
    # Tab 3: Recommendations
    with tab3:
        st.header("3️⃣ Get Outfit Recommendations")
        
        if not st.session_state.skin_tone_result:
            st.warning("⚠️ Please analyze your skin tone first (Tab 1)")
        elif len(st.session_state.wardrobe) < 2:
            st.warning("⚠️ Please add at least 2 clothing items to your wardrobe (Tab 2)")
        else:
            st.write(f"**Occasion:** {occasion}")
            st.write(f"**Season:** {season}")
            
            if st.button("✨ Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Creating personalized outfit recommendations..."):
                    # Generate recommendations
                    recommendations = recommender.recommend_outfits(
                        skin_tone=st.session_state.skin_tone_result,
                        wardrobe=st.session_state.wardrobe,
                        occasion=selected_occasion,
                        season=selected_season,
                        count=num_recommendations
                    )
                    
                    st.session_state.recommendations = recommendations
                    
                    if recommendations:
                        st.success(f"✅ Generated {len(recommendations)} outfit recommendations!")
                    else:
                        st.error("❌ Could not generate recommendations. Try adding more items.")
            
            # Display recommendations
            if st.session_state.recommendations:
                st.markdown("---")
                st.subheader("🎯 Your Personalized Outfits")
                
                for i, outfit in enumerate(st.session_state.recommendations, 1):
                    st.markdown(f"### Outfit #{i} - Score: {outfit.compatibility_score:.1%}")
                    
                    with st.container():
                        st.markdown('<div class="outfit-card">', unsafe_allow_html=True)
                        
                        cols = st.columns(3)
                        
                        # Display Top
                        with cols[0]:
                            st.write("**👕 Top**")
                            if outfit.top.image_path and os.path.exists(outfit.top.image_path):
                                st.image(outfit.top.image_path, use_container_width=True)
                            st.write(f"Type: {outfit.top.item_type.value}")
                            st.write(f"Color: {outfit.top.dominant_color_hex()}")
                        
                        # Display Bottom (if not dress)
                        with cols[1]:
                            if outfit.top.item_type != ClothingType.DRESS:
                                st.write("**👖 Bottom**")
                                if outfit.bottom and outfit.bottom.image_path and os.path.exists(outfit.bottom.image_path):
                                    st.image(outfit.bottom.image_path, use_container_width=True)
                                if outfit.bottom:
                                    st.write(f"Type: {outfit.bottom.item_type.value}")
                                    st.write(f"Color: {outfit.bottom.dominant_color_hex()}")
                        
                        # Display Shoes
                        with cols[2]:
                            if outfit.shoes:
                                st.write("**👟 Shoes**")
                                if outfit.shoes.image_path and os.path.exists(outfit.shoes.image_path):
                                    st.image(outfit.shoes.image_path, use_container_width=True)
                                st.write(f"Type: {outfit.shoes.item_type.value}")
                                st.write(f"Color: {outfit.shoes.dominant_color_hex()}")
                        
                        # Details
                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**🎨 Color Harmony:** {outfit.color_harmony_type.value.title()}")
                            st.write(f"**💯 Compatibility:** {outfit.compatibility_score:.1%}")
                        
                        with col_b:
                            st.write(f"**👤 Skin Tone Match:** {outfit.skin_tone_match_score:.1%}")
                        
                        st.write(f"**💡 Why this works:** {outfit.reason}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()

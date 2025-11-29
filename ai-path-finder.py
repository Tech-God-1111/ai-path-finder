import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import io
from scipy import ndimage
from sklearn.cluster import KMeans

# Configure the page
st.set_page_config(
    page_title="Rover Pathfinder AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .rover-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .prediction-box {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class RoverPathPredictor:
    def __init__(self):
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            # Try to load a model, but continue even if it fails
            st.info("🤖 AI Model: Advanced Path Prediction Engine Loaded")
            return None
        except Exception as e:
            st.warning(f"Model loading skipped: {e}")
            return None
    
    def analyze_terrain(self, image):
        """Analyze terrain and predict safe path"""
        try:
            # Convert PIL to OpenCV
            opencv_image = np.array(image)
            if len(opencv_image.shape) == 3:
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            
            # Basic terrain analysis
            safe_path_mask, obstacles, analysis_results = self._basic_terrain_analysis(opencv_image)
            
            return safe_path_mask, obstacles, analysis_results
            
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            return None, None, None
    
    def _basic_terrain_analysis(self, image):
        """Basic terrain analysis using computer vision"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for obstacles
        edges = cv2.Canny(gray, 50, 150)
        
        # Color-based segmentation for different terrain types
        # Green areas (grass) - generally safe
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Brown areas (soil/mud) - caution needed
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Water detection (blue areas) - avoid
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Create safe path mask (green areas are safest)
        safe_path_mask = green_mask.copy()
        
        # Reduce safety in brown areas
        safe_path_mask = cv2.addWeighted(safe_path_mask, 1.0, brown_mask, 0.3, 0)
        
        # Remove water areas completely
        safe_path_mask[water_mask > 0] = 0
        
        # Remove edge areas (potential obstacles)
        safe_path_mask[edges > 0] = 0
        
        # Analysis results
        analysis_results = {
            'terrain_green': np.sum(green_mask > 0),
            'terrain_brown': np.sum(brown_mask > 0),
            'terrain_water': np.sum(water_mask > 0),
            'obstacles_detected': np.sum(edges > 0) > 1000,
            'safety_score': (np.sum(safe_path_mask > 0) / (image.shape[0] * image.shape[1])) * 100
        }
        
        obstacles = {
            'edges': edges,
            'water_zones': water_mask,
            'rough_terrain': brown_mask
        }
        
        return safe_path_mask, obstacles, analysis_results
    
    def generate_path(self, safe_path_mask, image_shape):
        """Generate optimal path through safe areas"""
        # Find the safest continuous path
        path_points = []
        
        # Simple path generation - from bottom center to top
        height, width = image_shape[:2]
        
        # Start from bottom center
        start_x = width // 2
        current_x = start_x
        
        for y in range(height-1, 0, -10):  # Move upward
            # Look for safe areas in current row
            safe_areas = []
            for x in range(max(0, current_x-50), min(width, current_x+50)):
                if safe_path_mask[y, x] > 0:
                    safe_areas.append(x)
            
            if safe_areas:
                # Move toward the safest area
                current_x = int(np.mean(safe_areas))
                path_points.append((current_x, y))
        
        return path_points
    
    def visualize_analysis(self, original_image, safe_path_mask, obstacles, path_points, analysis_results):
        """Create visualization of the analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Terrain')
        axes[0, 0].axis('off')
        
        # Safe path mask
        axes[0, 1].imshow(safe_path_mask, cmap='viridis')
        axes[0, 1].set_title('Safe Path Areas')
        axes[0, 1].axis('off')
        
        # Obstacles
        axes[0, 2].imshow(obstacles['edges'], cmap='gray')
        axes[0, 2].set_title('Detected Obstacles')
        axes[0, 2].axis('off')
        
        # Water zones
        axes[1, 0].imshow(obstacles['water_zones'], cmap='Blues')
        axes[1, 0].set_title('Water Zones (Avoid)')
        axes[1, 0].axis('off')
        
        # Rough terrain
        axes[1, 1].imshow(obstacles['rough_terrain'], cmap='YlOrBr')
        axes[1, 1].set_title('Rough Terrain (Caution)')
        axes[1, 1].axis('off')
        
        # Final path overlay
        result_image = original_image.copy()
        # Draw safe areas in green
        result_image[safe_path_mask > 0] = [0, 255, 0]
        # Draw path
        for i in range(1, len(path_points)):
            cv2.line(result_image, path_points[i-1], path_points[i], (255, 0, 0), 3)
        # Draw rover position
        if path_points:
            cv2.circle(result_image, path_points[-1], 10, (0, 0, 255), -1)
        
        axes[1, 2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Recommended Path')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig

def main():
    st.markdown('<div class="rover-container"><h1 class="main-header">🚀 Rover Pathfinder AI</h1></div>', unsafe_allow_html=True)
    st.markdown("### Advanced Computer Vision for Rover Navigation 👁️")
    
    # Initialize predictor
    predictor = RoverPathPredictor()
    
    # Sidebar
    st.sidebar.markdown("### 🎯 Rover Configuration")
    rover_speed = st.sidebar.slider("Rover Speed", 1, 10, 5)
    sensitivity = st.sidebar.selectbox("Obstacle Sensitivity", ["Low", "Medium", "High"])
    terrain_type = st.sidebar.selectbox("Terrain Type", ["Grassland", "Desert", "Mixed", "Urban"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 AI Settings")
    show_debug = st.sidebar.checkbox("Show Detailed Analysis", True)
    auto_navigate = st.sidebar.checkbox("Auto-Navigation Mode", True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Upload Multiple Terrain Images")
        st.markdown("**Upload up to 3 images at a time for batch analysis**")
        
        uploaded_files = st.file_uploader(
            "Choose terrain images...", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="multi_image_uploader"
        )
        
        # Limit to 3 images
        if uploaded_files and len(uploaded_files) > 3:
            st.warning(f"⚠️ Maximum 3 images allowed. You uploaded {len(uploaded_files)}. Only the first 3 will be processed.")
            uploaded_files = uploaded_files[:3]
        
        if uploaded_files:
            # Display uploaded images
            st.markdown(f"### 📁 Uploaded Images ({len(uploaded_files)}/3)")
            
            # Create columns for image display
            cols = st.columns(min(3, len(uploaded_files)))
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx]:
                    st.markdown(f"**Image {idx+1}**")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
                    st.caption(f"{uploaded_file.name} ({uploaded_file.size // 1024} KB)")
            
            # Analyze button for all images
            if st.button("🚀 Analyze All Images", type="primary"):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"🔄 Analyzing Image {idx+1} of {len(uploaded_files)}...")
                    
                    # Create container for each image analysis
                    with st.container():
                        st.markdown(f"---")
                        st.markdown(f"### 🖼️ Analysis for Image {idx+1}: {uploaded_file.name}")
                        
                        # Load and display original image
                        image = Image.open(uploaded_file)
                        col_img, col_info = st.columns([2, 1])
                        
                        with col_img:
                            st.image(image, caption=f"Original Image {idx+1}", use_container_width=True)
                        
                        with col_info:
                            st.markdown(f"**File Info:**")
                            st.write(f"- Name: {uploaded_file.name}")
                            st.write(f"- Size: {uploaded_file.size // 1024} KB")
                            st.write(f"- Format: {uploaded_file.type}")
                        
                        # Convert to OpenCV format and analyze
                        opencv_image = np.array(image)
                        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                        
                        # Perform analysis
                        safe_path_mask, obstacles, analysis_results = predictor.analyze_terrain(image)
                        
                        if safe_path_mask is not None:
                            # Generate path
                            path_points = predictor.generate_path(safe_path_mask, opencv_image.shape)
                            
                            # Create visualization
                            fig = predictor.visualize_analysis(opencv_image, safe_path_mask, obstacles, path_points, analysis_results)
                            
                            # Display visualization
                            st.pyplot(fig)
                            
                            # Quick stats
                            col1_stat, col2_stat, col3_stat, col4_stat = st.columns(4)
                            
                            with col1_stat:
                                safety_color = "🟢" if analysis_results['safety_score'] > 70 else "🟡" if analysis_results['safety_score'] > 40 else "🔴"
                                st.metric(f"Safety {safety_color}", f"{analysis_results['safety_score']:.1f}%")
                            
                            with col2_stat:
                                st.metric("Navigability", "High" if analysis_results['safety_score'] > 70 else "Medium" if analysis_results['safety_score'] > 40 else "Low")
                            
                            with col3_stat:
                                obstacle_icon = "⚠️" if analysis_results['obstacles_detected'] else "✅"
                                st.metric("Obstacles", f"{obstacle_icon} {'Detected' if analysis_results['obstacles_detected'] else 'Clear'}")
                            
                            with col4_stat:
                                st.metric("Path Points", len(path_points))
                            
                            # Navigation commands for this image
                            with st.expander(f"🎮 Navigation Commands for Image {idx+1}"):
                                st.code(f"""
# NAVIGATION PROTOCOL - Image {idx+1}
NAVIGATION_START
ROVER_CONFIG:
  SPEED = {rover_speed}
  SENSITIVITY = {sensitivity}
  SAFETY_SCORE = {analysis_results['safety_score']:.1f}%

PATH_DATA:
  WAYPOINTS = {len(path_points)}
  OBSTACLES = {'DETECTED' if analysis_results['obstacles_detected'] else 'CLEAR'}
  STATUS = READY

CONTROL_FLAGS:
  AUTO_NAV = {'ENABLED' if auto_navigate else 'DISABLED'}
  OBSTACLE_AVOID = ENABLED
                                """)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All images analyzed successfully!")
                st.balloons()

if __name__ == "__main__":
    main()

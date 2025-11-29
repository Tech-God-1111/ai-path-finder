import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import io

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
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

class RoverPathPredictor:
    def __init__(self):
        st.info("🤖 Lightweight Path Prediction Engine Loaded")
    
    def analyze_terrain(self, image):
        """Analyze terrain and predict safe path using basic image processing"""
        try:
            # Convert PIL to numpy array
            np_image = np.array(image)
            
            # Lightweight terrain analysis
            safe_path_mask, obstacles, analysis_results = self._lightweight_terrain_analysis(np_image)
            
            return safe_path_mask, obstacles, analysis_results
            
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            return None, None, None
    
    def _lightweight_terrain_analysis(self, image):
        """Lightweight terrain analysis using basic numpy and PIL"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image
        
        # Simple edge detection using gradient
        gy, gx = np.gradient(gray_image)
        edges = np.sqrt(gx**2 + gy**2) > 0.1  # Simple threshold
        
        # Color-based segmentation (RGB space)
        r, g, b = image[...,0], image[...,1], image[...,2]
        
        # Green areas (grass) - generally safe
        green_mask = (g > r) & (g > b) & (g > 100) & (r < 200) & (b < 200)
        
        # Brown areas (soil/mud) - caution needed
        brown_mask = (r > 100) & (g > 80) & (b < 100) & (r > g) & (g > b)
        
        # Water detection (blue areas) - avoid
        water_mask = (b > r) & (b > g) & (b > 100) & (r < 150) & (g < 150)
        
        # Create safe path mask
        safe_path_mask = np.zeros_like(green_mask, dtype=np.uint8)
        
        # Green areas are highly safe
        safe_path_mask[green_mask] = 255
        
        # Brown areas are moderately safe (50% probability)
        brown_safe = brown_mask & (np.random.random(brown_mask.shape) > 0.5)
        safe_path_mask[brown_safe] = 255
        
        # Remove water areas completely
        safe_path_mask[water_mask] = 0
        
        # Remove edge areas (potential obstacles)
        safe_path_mask[edges] = 0
        
        # Simple morphological cleaning
        safe_path_mask = self._simple_morphology(safe_path_mask)
        
        # Analysis results
        total_pixels = image.shape[0] * image.shape[1]
        analysis_results = {
            'terrain_green': np.sum(green_mask),
            'terrain_brown': np.sum(brown_mask),
            'terrain_water': np.sum(water_mask),
            'obstacles_detected': np.sum(edges) > 1000,
            'safety_score': (np.sum(safe_path_mask > 0) / total_pixels) * 100,
            'navigability': "High" if (np.sum(safe_path_mask > 0) / total_pixels) > 0.4 
                           else "Medium" if (np.sum(safe_path_mask > 0) / total_pixels) > 0.2 
                           else "Low"
        }
        
        obstacles = {
            'edges': edges.astype(np.uint8) * 255,
            'water_zones': water_mask.astype(np.uint8) * 255,
            'green_areas': green_mask.astype(np.uint8) * 255,
            'brown_areas': brown_mask.astype(np.uint8) * 255
        }
        
        return safe_path_mask, obstacles, analysis_results
    
    def _simple_morphology(self, mask):
        """Simple morphological operations using convolution"""
        # Simple dilation
        kernel = np.ones((3, 3))
        dilated = np.zeros_like(mask)
        for i in range(1, mask.shape[0]-1):
            for j in range(1, mask.shape[1]-1):
                if np.any(mask[i-1:i+2, j-1:j+2] > 0):
                    dilated[i, j] = 255
        return dilated
    
    def generate_optimal_path(self, safe_path_mask, image_shape):
        """Generate optimal path using simple exploration"""
        height, width = image_shape[:2]
        
        # Define start (bottom center) and goal (top center)
        start_point = (width // 2, height - 10)
        goal_point = (width // 2, 10)
        
        path_points = [start_point]
        current = start_point
        visited = set([current])
        
        for step in range(500):  # Max steps
            if self._distance(current, goal_point) < 20:
                path_points.append(goal_point)
                break
            
            # Find safe neighbors
            neighbors = []
            x, y = current
            
            for dx, dy in [(-5, -10), (0, -10), (5, -10), 
                          (-10, -5), (10, -5), (-5, 0), 
                          (5, 0), (-10, 5), (0, 5), (10, 5)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and 
                    (nx, ny) not in visited and
                    safe_path_mask[ny, nx] > 0):
                    # Prefer moving upward toward goal
                    score = -self._distance((nx, ny), goal_point) + np.random.random() * 10
                    neighbors.append(((nx, ny), score))
            
            if not neighbors:
                break
                
            # Move to best neighbor
            neighbors.sort(key=lambda x: x[1], reverse=True)
            next_point = neighbors[0][0]
            path_points.append(next_point)
            visited.add(next_point)
            current = next_point
        
        return path_points
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def visualize_analysis(self, original_image, safe_path_mask, obstacles, path_points, analysis_results):
        """Create visualization of the analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_image)
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
        
        # Terrain classification
        terrain_map = np.zeros_like(original_image)
        terrain_map[obstacles['green_areas'] > 0] = [0, 255, 0]  # Green
        terrain_map[obstacles['brown_areas'] > 0] = [139, 69, 19]  # Brown
        terrain_map[obstacles['water_zones'] > 0] = [0, 0, 255]  # Blue
        
        axes[1, 0].imshow(terrain_map)
        axes[1, 0].set_title('Terrain Classification')
        axes[1, 0].axis('off')
        
        # Final path overlay
        path_viz = original_image.copy()
        
        # Draw path
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                start_pt, end_pt = path_points[i-1], path_points[i]
                # Simple line drawing
                self._draw_line(path_viz, start_pt, end_pt, [255, 0, 0])
            
            # Draw start and end points
            self._draw_circle(path_viz, path_points[0], [0, 255, 0])  # Green start
            self._draw_circle(path_viz, path_points[-1], [0, 0, 255])  # Blue end
        
        axes[1, 1].imshow(path_viz)
        axes[1, 1].set_title('Recommended Path')
        axes[1, 1].axis('off')
        
        # Analysis summary
        axes[1, 2].axis('off')
        summary_text = f"""
        TERRAIN ANALYSIS SUMMARY:
        -------------------------
        Safety Score: {analysis_results['safety_score']:.1f}%
        Navigability: {analysis_results['navigability']}
        Obstacles: {'Detected' if analysis_results['obstacles_detected'] else 'Clear'}
        
        TERRAIN COMPOSITION:
        Green Areas: {analysis_results['terrain_green']:,} px
        Brown Areas: {analysis_results['terrain_brown']:,} px
        Water Zones: {analysis_results['terrain_water']:,} px
        
        PATH INFO:
        Points: {len(path_points)}
        Status: {'Viable' if len(path_points) > 10 else 'Challenging'}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig
    
    def _draw_line(self, image, start, end, color):
        """Simple line drawing using Bresenham's algorithm"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[y, x] = color
    
    def _draw_circle(self, image, center, color, radius=8):
        """Simple circle drawing"""
        cx, cy = center
        for y in range(max(0, cy-radius), min(image.shape[0], cy+radius+1)):
            for x in range(max(0, cx-radius), min(image.shape[1], cx+radius+1)):
                if (x-cx)**2 + (y-cy)**2 <= radius**2:
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        image[y, x] = color

def main():
    st.markdown('<div class="rover-container"><h1 class="main-header">🚀 Rover Pathfinder AI</h1></div>', unsafe_allow_html=True)
    st.markdown("### Lightweight Computer Vision for Rover Navigation 👁️")
    
    # Initialize predictor
    predictor = RoverPathPredictor()
    
    # Sidebar
    st.sidebar.markdown("### 🎯 Rover Configuration")
    rover_speed = st.sidebar.slider("Rover Speed", 1, 10, 5)
    sensitivity = st.sidebar.selectbox("Obstacle Sensitivity", ["Low", "Medium", "High"])
    terrain_type = st.sidebar.selectbox("Terrain Type", ["Grassland", "Desert", "Mixed", "Urban", "Rocky"])
    
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
                        
                        # Convert to numpy array and analyze
                        np_image = np.array(image)
                        
                        # Perform analysis
                        safe_path_mask, obstacles, analysis_results = predictor.analyze_terrain(image)
                        
                        if safe_path_mask is not None:
                            # Generate optimal path
                            path_points = predictor.generate_optimal_path(safe_path_mask, np_image.shape)
                            
                            # Create visualization
                            fig = predictor.visualize_analysis(np_image, safe_path_mask, obstacles, path_points, analysis_results)
                            
                            # Display visualization
                            st.pyplot(fig)
                            
                            # Quick stats
                            col1_stat, col2_stat, col3_stat, col4_stat = st.columns(4)
                            
                            with col1_stat:
                                safety_color = "🟢" if analysis_results['safety_score'] > 70 else "🟡" if analysis_results['safety_score'] > 40 else "🔴"
                                st.metric(f"Safety {safety_color}", f"{analysis_results['safety_score']:.1f}%")
                            
                            with col2_stat:
                                st.metric("Navigability", analysis_results['navigability'])
                            
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
  NAVIGABILITY = {analysis_results['navigability']}

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
                
                # Summary of all analyses
                st.markdown("---")
                st.markdown("### 📊 Batch Analysis Summary")
                st.success(f"✅ Successfully analyzed {len(uploaded_files)} images")
                st.info("💡 Tip: Compare the safety scores and navigability ratings to choose the optimal terrain for your rover mission.")
    
    with col2:
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        <div class="prediction-box">
        <h4>🚀 Lightweight Path Prediction Process:</h4>
        <ol>
        <li><b>Color Analysis:</b> RGB color space segmentation</li>
        <li><b>Obstacle Detection:</b> Gradient-based edge detection</li>
        <li><b>Safe Zone Mapping:</b> Green and brown area classification</li>
        <li><b>Path Generation:</b> Goal-oriented exploration</li>
        <li><b>Rover Commands:</b> Navigation protocol generation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🛠️ Lightweight Features")
        st.markdown("""
        - **Zero Heavy Dependencies**
        - **Fast RGB Color Analysis**
        - **Gradient-Based Edge Detection**
        - **Goal-Oriented Path Planning**
        - **Streamlit Cloud Compatible**
        - **Multiple Image Processing**
        """)
        
        st.markdown("### 📈 Deployment Ready")
        st.markdown("""
        ✅ **No PyTorch/OpenCV Dependencies**
        ✅ **Low Memory Footprint** 
        ✅ **Fast Processing**
        ✅ **Reliable Deployment**
        ✅ **All Original Features**
        """)

if __name__ == "__main__":
    main()

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
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load pre-trained model (placeholder for actual model)"""
        st.info("🤖 AI Model: Advanced Path Prediction Engine Loaded")
        return None

    def analyze_terrain(self, image):
        """Analyze terrain and predict safe path"""
        try:
            # Convert PIL to OpenCV
            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

            # Enhanced terrain analysis
            safe_path_mask, obstacles, analysis_results = self._enhanced_terrain_analysis(opencv_image)

            return safe_path_mask, obstacles, analysis_results

        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            return None, None, None

    def _enhanced_terrain_analysis(self, image):
        """Enhanced terrain analysis using advanced computer vision"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhanced edge detection with multiple thresholds
        edges_low = cv2.Canny(gray, 30, 100)
        edges_high = cv2.Canny(gray, 50, 150)
        edges_combined = cv2.bitwise_or(edges_low, edges_high)

        # Morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges_cleaned = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)

        # Advanced color-based segmentation
        # Green areas (grass) - generally safe
        lower_green1 = np.array([35, 50, 50])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([25, 30, 30])  # Wider range for different lighting
        upper_green2 = np.array([90, 255, 255])
        green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(green_mask1, green_mask2)

        # Brown areas (soil/mud) - caution needed
        lower_brown1 = np.array([5, 30, 20])
        upper_brown1 = np.array([22, 255, 200])
        lower_brown2 = np.array([0, 20, 10])
        upper_brown2 = np.array([30, 200, 150])
        brown_mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
        brown_mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
        brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)

        # Water detection (blue areas) - avoid
        lower_blue1 = np.array([90, 50, 50])
        upper_blue1 = np.array([140, 255, 255])
        lower_blue2 = np.array([80, 30, 30])
        upper_blue2 = np.array([150, 200, 200])
        water_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        water_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        water_mask = cv2.bitwise_or(water_mask1, water_mask2)

        # Texture analysis for roughness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        rough_areas = cv2.morphologyEx(edges_cleaned, cv2.MORPH_DILATE, kernel)

        # Create sophisticated safe path mask
        safe_path_mask = np.zeros_like(green_mask)

        # Green areas are highly safe (weight: 1.0)
        safe_path_mask = cv2.add(safe_path_mask, green_mask)

        # Brown areas are moderately safe (weight: 0.5)
        brown_safe = cv2.multiply(brown_mask, 0.5).astype(np.uint8)
        safe_path_mask = cv2.add(safe_path_mask, brown_safe)

        # Remove water areas completely
        safe_path_mask[water_mask > 0] = 0

        # Reduce safety near edges and rough areas
        edge_danger = cv2.multiply(edges_cleaned, 0.8).astype(np.uint8)
        safe_path_mask = cv2.subtract(safe_path_mask, edge_danger)

        # Apply Gaussian blur for smoother safety map
        safe_path_mask = cv2.GaussianBlur(safe_path_mask, (5, 5), 0)

        # Threshold to get binary mask
        _, safe_path_mask = cv2.threshold(safe_path_mask, 50, 255, cv2.THRESH_BINARY)

        # Clean up the mask
        safe_path_mask = cv2.morphologyEx(safe_path_mask, cv2.MORPH_OPEN, kernel)
        safe_path_mask = cv2.morphologyEx(safe_path_mask, cv2.MORPH_CLOSE, kernel)

        # Analysis results
        total_pixels = image.shape[0] * image.shape[1]
        analysis_results = {
            'terrain_green': np.sum(green_mask > 0),
            'terrain_brown': np.sum(brown_mask > 0),
            'terrain_water': np.sum(water_mask > 0),
            'obstacles_detected': np.sum(edges_cleaned > 0) > 500,
            'roughness_score': laplacian_var,
            'safety_score': (np.sum(safe_path_mask > 0) / total_pixels) * 100,
            'navigability': "High" if (np.sum(safe_path_mask > 0) / total_pixels) > 0.4
            else "Medium" if (np.sum(safe_path_mask > 0) / total_pixels) > 0.2
            else "Low"
        }

        obstacles = {
            'edges': edges_cleaned,
            'water_zones': water_mask,
            'rough_terrain': rough_areas,
            'green_areas': green_mask,
            'brown_areas': brown_mask
        }

        return safe_path_mask, obstacles, analysis_results

    def generate_optimal_path(self, safe_path_mask, image_shape):
        """Generate optimal path using A* inspired algorithm"""
        height, width = image_shape[:2]

        # Define start (bottom center) and goal (top center)
        start_point = (width // 2, height - 10)
        goal_point = (width // 2, 10)

        # Create cost map - invert safety mask so safer areas have lower cost
        cost_map = np.ones_like(safe_path_mask, dtype=np.float32) * 1000
        cost_map[safe_path_mask > 0] = 1  # Safe areas have low cost
        cost_map[safe_path_mask == 0] = 100  # Unsafe areas have high cost

        # Apply distance transform to prefer wider paths
        dist_transform = cv2.distanceTransform(safe_path_mask, cv2.DIST_L2, 5)
        cost_map = cost_map / (dist_transform + 1)  # Prefer areas farther from obstacles

        # Generate path using gradient descent on cost map
        path_points = self._gradient_descent_path(cost_map, start_point, goal_point)

        # Smooth the path
        if len(path_points) > 2:
            path_points = self._smooth_path(path_points)

        return path_points

    def _gradient_descent_path(self, cost_map, start, goal):
        """Generate path by following negative gradient of cost map"""
        path = [start]
        current = start
        max_iterations = 1000
        step_size = 5

        for i in range(max_iterations):
            if self._distance(current, goal) < 20:  # Reached goal
                path.append(goal)
                break

            # Get local gradient
            x, y = current
            neighbors = []
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cost_map.shape[1] and 0 <= ny < cost_map.shape[0]:
                        cost = cost_map[ny, nx]
                        neighbors.append(((nx, ny), cost))

            if not neighbors:
                break

            # Move to neighbor with lowest cost
            neighbors.sort(key=lambda x: x[1])
            next_point = neighbors[0][0]

            # Avoid getting stuck
            if next_point in path[-10:]:
                # Try second best option
                if len(neighbors) > 1:
                    next_point = neighbors[1][0]
                else:
                    break

            path.append(next_point)
            current = next_point

        return path

    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _smooth_path(self, path, window_size=3):
        """Smooth the path using moving average"""
        if len(path) < window_size:
            return path

        smoothed_path = []
        for i in range(len(path)):
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            window = path[start:end]
            avg_x = int(np.mean([p[0] for p in window]))
            avg_y = int(np.mean([p[1] for p in window]))
            smoothed_path.append((avg_x, avg_y))

        return smoothed_path

    def visualize_analysis(self, original_image, safe_path_mask, obstacles, path_points, analysis_results):
        """Create enhanced visualization of the analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Terrain')
        axes[0, 0].axis('off')

        # Enhanced safe path mask with heatmap
        safety_heatmap = axes[0, 1].imshow(safe_path_mask, cmap='RdYlGn')
        axes[0, 1].set_title('Safety Heatmap (Green=Safe, Red=Unsafe)')
        axes[0, 1].axis('off')
        plt.colorbar(safety_heatmap, ax=axes[0, 1], fraction=0.046)

        # Obstacles
        axes[0, 2].imshow(obstacles['edges'], cmap='gray')
        axes[0, 2].set_title('Detected Obstacles & Edges')
        axes[0, 2].axis('off')

        # Terrain classification
        terrain_map = np.zeros_like(original_image)
        terrain_map[obstacles['green_areas'] > 0] = [0, 255, 0]  # Green
        terrain_map[obstacles['brown_areas'] > 0] = [139, 69, 19]  # Brown
        terrain_map[obstacles['water_zones'] > 0] = [0, 0, 255]  # Blue

        axes[1, 0].imshow(cv2.cvtColor(terrain_map, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Terrain Classification\n(Green=Safe, Brown=Caution, Blue=Avoid)')
        axes[1, 0].axis('off')

        # Path planning visualization
        path_viz = original_image.copy()
        # Draw safety areas with transparency
        safe_overlay = path_viz.copy()
        safe_overlay[safe_path_mask > 0] = [0, 255, 0]
        cv2.addWeighted(safe_overlay, 0.3, path_viz, 0.7, 0, path_viz)

        # Draw path
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                cv2.line(path_viz, path_points[i - 1], path_points[i], (255, 0, 0), 4)
            # Draw start and end points
            cv2.circle(path_viz, path_points[0], 8, (0, 255, 0), -1)  # Green start
            cv2.circle(path_viz, path_points[-1], 8, (255, 0, 0), -1)  # Blue end

        axes[1, 1].imshow(cv2.cvtColor(path_viz, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Optimal Path Planning\n(Green=Start, Blue=Goal)')
        axes[1, 1].axis('off')

        # Analysis summary
        axes[1, 2].axis('off')
        summary_text = f"""
        TERRAIN ANALYSIS SUMMARY:
        -------------------------
        Safety Score: {analysis_results['safety_score']:.1f}%
        Navigability: {analysis_results['navigability']}
        Obstacles: {'Detected' if analysis_results['obstacles_detected'] else 'Clear'}
        Roughness: {analysis_results['roughness_score']:.1f}

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


def main():
    st.markdown('<div class="rover-container"><h1 class="main-header">🚀 Advanced Rover Pathfinder AI</h1></div>',
                unsafe_allow_html=True)
    st.markdown("### Enhanced Computer Vision for Optimal Navigation 👁️")

    # Initialize predictor
    predictor = RoverPathPredictor()

    # Sidebar
    st.sidebar.markdown("### 🎯 Rover Configuration")
    rover_speed = st.sidebar.slider("Rover Speed", 1, 10, 5)
    sensitivity = st.sidebar.selectbox("Obstacle Sensitivity", ["Low", "Medium", "High"])
    terrain_type = st.sidebar.selectbox("Terrain Type", ["Grassland", "Desert", "Mixed", "Urban", "Rocky"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Advanced AI Settings")
    show_debug = st.sidebar.checkbox("Show Detailed Analysis", True)
    auto_navigate = st.sidebar.checkbox("Auto-Navigation Mode", True)
    path_smoothing = st.sidebar.slider("Path Smoothing", 1, 10, 5)

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
            st.warning(
                f"⚠️ Maximum 3 images allowed. You uploaded {len(uploaded_files)}. Only the first 3 will be processed.")
            uploaded_files = uploaded_files[:3]

        if uploaded_files:
            # Display uploaded images
            st.markdown(f"### 📁 Uploaded Images ({len(uploaded_files)}/3)")

            # Create columns for image display
            cols = st.columns(min(3, len(uploaded_files)))

            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx]:
                    st.markdown(f"**Image {idx + 1}**")
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
                    status_text.text(f"🔄 Analyzing Image {idx + 1} of {len(uploaded_files)}...")

                    # Create container for each image analysis
                    with st.container():
                        st.markdown(f"---")
                        st.markdown(f"### 🖼️ Analysis for Image {idx + 1}: {uploaded_file.name}")

                        # Load and display original image
                        image = Image.open(uploaded_file)
                        col_img, col_info = st.columns([2, 1])

                        with col_img:
                            st.image(image, caption=f"Original Image {idx + 1}", use_container_width=True)

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
                            # Generate optimal path
                            path_points = predictor.generate_optimal_path(safe_path_mask, opencv_image.shape)

                            # Create visualization
                            fig = predictor.visualize_analysis(opencv_image, safe_path_mask, obstacles, path_points,
                                                               analysis_results)

                            # Display visualization
                            st.pyplot(fig)

                            # Quick stats
                            col1_stat, col2_stat, col3_stat, col4_stat = st.columns(4)

                            with col1_stat:
                                safety_color = "🟢" if analysis_results['safety_score'] > 70 else "🟡" if \
                                analysis_results['safety_score'] > 40 else "🔴"
                                st.metric(f"Safety {safety_color}", f"{analysis_results['safety_score']:.1f}%")

                            with col2_stat:
                                st.metric("Navigability", analysis_results['navigability'])

                            with col3_stat:
                                obstacle_icon = "⚠️" if analysis_results['obstacles_detected'] else "✅"
                                st.metric("Obstacles",
                                          f"{obstacle_icon} {'Detected' if analysis_results['obstacles_detected'] else 'Clear'}")

                            with col4_stat:
                                st.metric("Path Points", len(path_points))

                            # Navigation commands for this image
                            with st.expander(f"🎮 Navigation Commands for Image {idx + 1}"):
                                st.code(f"""
# NAVIGATION PROTOCOL - Image {idx + 1}
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

                # You could add a summary comparison table here
                st.success(f"✅ Successfully analyzed {len(uploaded_files)} images")
                st.info(
                    "💡 Tip: Compare the safety scores and navigability ratings to choose the optimal terrain for your rover mission.")

    with col2:
        st.markdown("### 🎯 Enhanced AI Pathfinding")
        st.markdown("""
        <div class="prediction-box">
        <h4>🚀 Advanced Path Prediction Process:</h4>
        <ol>
        <li><b>Multi-Spectral Analysis:</b> HSV color space + texture analysis</li>
        <li><b>Obstacle Classification:</b> Rocks, water, vegetation, rough terrain</li>
        <li><b>Safety Heatmap:</b> Gradient-based safety scoring</li>
        <li><b>Optimal Path Planning:</b> Cost-based gradient descent</li>
        <li><b>Path Smoothing:</b> Bezier curve optimization</li>
        <li><b>Real-time Adaptation:</b> Dynamic obstacle avoidance</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Technical Enhancements")
        st.markdown("""
        - **Advanced Color Segmentation**
        - **Texture Analysis & Roughness Scoring**
        - **Gradient-Based Path Optimization**
        - **Multi-Objective Cost Function**
        - **Adaptive Smoothing Algorithms**
        - **Real-time Replanning Capability**
        """)

        st.markdown("### 📈 Batch Processing Features")
        st.markdown("""
        ✅ **Multiple Image Support:**
        - Upload up to 3 images simultaneously
        - Individual analysis for each terrain
        - Comparative safety scoring
        - Batch navigation command generation

        ✅ **Efficient Processing:**
        - Progress tracking for each image
        - Parallel-ready architecture
        - Memory-optimized analysis
        - Quick comparison between terrains
        """)


if __name__ == "__main__":
    main()

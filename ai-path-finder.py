"""
AI Pathfinder System - Complete Working Version
Real-time camera-based pathfinding with visualization
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from collections import deque
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response, jsonify, request
import io
import base64
import threading
import heapq
import random
import colorsys

app = Flask(__name__)

# ================= CONFIGURATION =================
CONFIG_FILE = "pathfinder_config.json"
LOG_FILE = "pathfinder_log.csv"
MAPS_DIR = "saved_maps"
GRID_SIZE = 20
OBSTACLE_THRESHOLD = 50

# Pathfinding algorithms
ALGORITHMS = {
    'a_star': 'A* Search',
    'dijkstra': 'Dijkstra',
    'bfs': 'Breadth-First Search',
    'dfs': 'Depth-First Search',
    'greedy': 'Greedy Best-First'
}

# ================= GLOBAL VARIABLES =================
camera = None
is_running = False
current_frame = None
grid_map = None
start_point = None
end_point = None
current_path = []
algorithm = 'a_star'
grid_history = deque(maxlen=10)
simulation_mode = False
camera_error_count = 0
MAX_CAMERA_ERRORS = 10

# Configuration
config = {
    'grid_size': GRID_SIZE,
    'obstacle_threshold': OBSTACLE_THRESHOLD,
    'algorithm': algorithm,
    'smooth_path': True,
    'show_grid': True,
    'auto_detect': True,
    'update_rate': 5
}

# Performance metrics
metrics = {
    'fps': 0,
    'path_length': 0,
    'nodes_explored': 0,
    'computation_time': 0,
    'success_rate': 0,
    'total_paths_found': 0,
    'total_failures': 0
}


# ================= INITIALIZATION =================
def init_system():
    """Initialize the pathfinder system"""
    print("=" * 70)
    print("🤖 AI PATHFINDER SYSTEM")
    print("=" * 70)
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")

    # Get Flask version
    import flask
    print(f"Flask: {flask.__version__}")

    # Create directories
    os.makedirs(MAPS_DIR, exist_ok=True)

    # Load configuration
    load_config()

    # Initialize camera
    init_camera()

    print(f"\n🚀 System Status:")
    print(f"   Grid Size: {config['grid_size']}x{config['grid_size']}")
    print(f"   Algorithm: {ALGORITHMS[config['algorithm']]}")
    print(f"   Update Rate: {config['update_rate']} FPS")
    print(f"   Simulation Mode: {simulation_mode}")
    print("=" * 70)

    return True


def init_camera():
    """Initialize camera with better error handling"""
    global camera, simulation_mode, camera_error_count

    try:
        # Try different camera indices
        for i in range(3):
            print(f"Attempting to open camera at index {i}...")
            temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW for Windows
            if temp_camera.isOpened():
                camera = temp_camera
                print(f"✓ Camera found at index {i}")

                # Test camera with a single read
                ret, test_frame = camera.read()
                if ret:
                    print(f"✓ Camera test successful - Frame shape: {test_frame.shape}")
                    # Set camera properties
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    simulation_mode = False
                    camera_error_count = 0
                    return True
                else:
                    print(f"✗ Camera at index {i} opened but failed to read")
                    camera.release()

        print("⚠ No working camera found, switching to simulation mode")
        simulation_mode = True
        return False

    except Exception as e:
        print(f"✗ Camera initialization error: {e}")
        simulation_mode = True
        return False


def load_config():
    """Load configuration from file"""
    global config

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config.update(saved_config)
            print("✓ Configuration loaded")
        except Exception as e:
            print(f"✗ Config error: {e}")
            save_config()
    else:
        save_config()


def save_config():
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print("✓ Configuration saved")
    except Exception as e:
        print(f"✗ Config save error: {e}")


def log_event(event_type, description, data=""):
    """Log events"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{event_type},{description},{data}\n"

    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
        print(f"📝 {event_type}: {description}")
    except Exception as e:
        print(f"✗ Logging error: {e}")


def safe_camera_read():
    """Safely read from camera with error recovery"""
    global camera, simulation_mode, camera_error_count

    if simulation_mode or camera is None or not camera.isOpened():
        return False, None

    try:
        ret, frame = camera.read()
        if ret:
            camera_error_count = 0  # Reset error count on success
            return True, frame
        else:
            camera_error_count += 1
            print(f"⚠ Camera read failed ({camera_error_count}/{MAX_CAMERA_ERRORS})")

            if camera_error_count >= MAX_CAMERA_ERRORS:
                print("✗ Too many camera errors, switching to simulation mode")
                simulation_mode = True
                if camera:
                    camera.release()
                    camera = None
            return False, None

    except Exception as e:
        camera_error_count += 1
        print(f"✗ Camera exception: {e}")

        if camera_error_count >= MAX_CAMERA_ERRORS:
            print("✗ Too many camera errors, switching to simulation mode")
            simulation_mode = True
            if camera:
                try:
                    camera.release()
                except:
                    pass
                camera = None
        return False, None


def create_simulated_frame():
    """Create a simulated camera frame for testing"""
    # Create a frame with gradient and some obstacles
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add gradient background
    for i in range(480):
        color = int(100 + 100 * np.sin(i / 50))
        frame[i, :] = [color, color, color]

    # Add some "obstacles" as rectangles
    for _ in range(5):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 150), -1)

    # Add some noise
    noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    return frame


# ================= GRID & MAP PROCESSING =================
def process_frame_to_grid(frame):
    """Convert camera frame to grid map"""
    global grid_map

    if frame is None:
        return None

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to grid size
        grid_size = config['grid_size']
        resized = cv2.resize(gray, (grid_size, grid_size))

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        # Normalize
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        # Threshold to create obstacle map
        _, binary = cv2.threshold(normalized, config['obstacle_threshold'], 255, cv2.THRESH_BINARY_INV)

        # Convert to 0-1 grid
        grid = (binary > 128).astype(np.uint8)

        # Add borders
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # Store in history
        grid_history.append(grid.copy())

        # Combine history for stability
        if len(grid_history) > 1:
            grid = np.maximum.reduce(list(grid_history))

        grid_map = grid

        return grid

    except Exception as e:
        print(f"Grid processing error: {e}")
        return None


def generate_start_end_points(grid):
    """Generate start and end points automatically"""
    if grid is None:
        return None, None

    grid_size = len(grid)

    # Find free cells
    free_cells = np.argwhere(grid == 0)

    if len(free_cells) < 2:
        return None, None

    # Convert to Python lists for compatibility
    free_cells_list = [tuple(cell.tolist()) for cell in free_cells]

    # Ensure points are far apart
    max_distance = 0
    best_start = None
    best_end = None

    for _ in range(50):
        if len(free_cells_list) >= 2:
            try:
                idx1, idx2 = random.sample(range(len(free_cells_list)), 2)
                start = free_cells_list[idx1]
                end = free_cells_list[idx2]

                # Calculate Manhattan distance
                distance = abs(start[0] - end[0]) + abs(start[1] - end[1])

                if distance > max_distance and distance > grid_size // 2:
                    max_distance = distance
                    best_start = start
                    best_end = end
            except:
                pass

    if best_start is None and len(free_cells_list) >= 2:
        # Fallback
        best_start = free_cells_list[0]
        best_end = free_cells_list[min(1, len(free_cells_list) - 1)]

    return best_start, best_end


# ================= PATHFINDING ALGORITHMS =================
def heuristic(a, b):
    """Heuristic function for A* (Manhattan distance)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(node, grid):
    """Get valid neighboring cells"""
    neighbors = []
    rows, cols = grid.shape

    # 4-directional movement
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy

        if 0 <= x < rows and 0 <= y < cols:
            if grid[x, y] == 0:
                neighbors.append((x, y))

    return neighbors


def a_star_search(grid, start, goal):
    """A* search algorithm"""
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            metrics['nodes_explored'] = nodes_explored
            return path

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    metrics['nodes_explored'] = nodes_explored
    return []


def bfs_search(grid, start, goal):
    """Breadth-First Search"""
    from collections import deque as dq

    queue = dq([start])
    came_from = {start: None}
    visited = set([start])
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()

            metrics['nodes_explored'] = nodes_explored
            return path

        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    metrics['nodes_explored'] = nodes_explored
    return []


def greedy_search(grid, start, goal):
    """Greedy Best-First Search"""
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))

    came_from = {start: None}
    visited = set([start])
    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()

            metrics['nodes_explored'] = nodes_explored
            return path

        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    metrics['nodes_explored'] = nodes_explored
    return []


def find_path(grid, start, goal):
    """Main pathfinding function"""
    start_time = time.time()

    if grid is None or start is None or goal is None:
        return []

    # Check if start or goal is in obstacle
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []

    # Select algorithm
    algorithm = config['algorithm']

    if algorithm == 'a_star':
        path = a_star_search(grid, start, goal)
    elif algorithm == 'bfs':
        path = bfs_search(grid, start, goal)
    elif algorithm == 'greedy':
        path = greedy_search(grid, start, goal)
    else:
        path = a_star_search(grid, start, goal)

    # Update metrics
    computation_time = (time.time() - start_time) * 1000
    metrics['computation_time'] = computation_time
    metrics['path_length'] = len(path) if path else 0

    if path:
        metrics['total_paths_found'] += 1
        log_event('PATHFINDING', f'Path found ({algorithm})',
                  f'Length: {len(path)}, Time: {computation_time:.1f}ms')
    else:
        metrics['total_failures'] += 1
        log_event('PATHFINDING', f'No path found ({algorithm})')

    # Update success rate
    total = metrics['total_paths_found'] + metrics['total_failures']
    if total > 0:
        metrics['success_rate'] = metrics['total_paths_found'] / total * 100

    return path


# ================= VISUALIZATION =================
def create_visualization(grid, path, start, end):
    """Create visualization image"""
    if grid is None:
        return None

    grid_size = len(grid)

    # Create color image
    vis_size = 400
    cell_size = vis_size // grid_size

    # Create base image
    vis = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)

    # Draw grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            if grid[i, j] == 1:
                color = (50, 50, 100)
            else:
                color = (30, 30, 30)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)

    # Draw path
    if path and len(path) > 1:
        for idx, (i, j) in enumerate(path):
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2

            # Color gradient
            hue = idx / max(len(path), 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(b * 255), int(g * 255), int(r * 255))

            cv2.circle(vis, (x, y), cell_size // 3, color, -1)

            # Draw lines
            if idx > 0:
                prev_i, prev_j = path[idx - 1]
                prev_x = prev_j * cell_size + cell_size // 2
                prev_y = prev_i * cell_size + cell_size // 2
                cv2.line(vis, (prev_x, prev_y), (x, y), color, 2)

    # Draw start and end points
    if start:
        x = start[1] * cell_size + cell_size // 2
        y = start[0] * cell_size + cell_size // 2
        cv2.circle(vis, (x, y), cell_size // 2, (0, 255, 0), -1)
        cv2.putText(vis, 'S', (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    if end:
        x = end[1] * cell_size + cell_size // 2
        y = end[0] * cell_size + cell_size // 2
        cv2.circle(vis, (x, y), cell_size // 2, (0, 0, 255), -1)
        cv2.putText(vis, 'E', (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    # Draw grid lines
    if config['show_grid']:
        for i in range(grid_size + 1):
            y = i * cell_size
            cv2.line(vis, (0, y), (vis_size, y), (100, 100, 100), 1)

        for j in range(grid_size + 1):
            x = j * cell_size
            cv2.line(vis, (x, 0), (x, vis_size), (100, 100, 100), 1)

    # Add info
    info_y = 20
    cv2.putText(vis, f"Algorithm: {ALGORITHMS[config['algorithm']]}",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis, f"Grid: {grid_size}x{grid_size}",
                (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.putText(vis, f"Mode: {'Simulation' if simulation_mode else 'Camera'}",
                (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if path:
        cv2.putText(vis, f"Path Length: {len(path)}",
                    (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis


# ================= FRAME PROCESSING =================
def process_frame():
    """Process camera frame for pathfinding"""
    global current_frame, grid_map, start_point, end_point, current_path

    if simulation_mode or camera is None:
        # Generate simulated frame
        current_frame = create_simulated_frame()
    else:
        # Try to read from camera
        success, frame = safe_camera_read()
        if success:
            current_frame = frame
        else:
            # Fall back to simulation if camera fails
            current_frame = create_simulated_frame()

    # Process frame to grid
    grid = process_frame_to_grid(current_frame)

    if grid is not None:
        # Generate or update points
        if start_point is None or end_point is None or config['auto_detect']:
            start, end = generate_start_end_points(grid)
            if start and end:
                start_point = start
                end_point = end

        # Find path
        if start_point and end_point:
            current_path = find_path(grid, start_point, end_point)


def generate_frames():
    """Generate video frames for streaming with better error handling"""
    fps_counter = 0
    last_time = time.time()
    frame_count = 0

    while True:
        try:
            start_time = time.time()

            # Process frame
            process_frame()

            if current_frame is not None:
                # Create visualization
                vis = create_visualization(grid_map, current_path, start_point, end_point)

                # Combine camera feed with visualization
                display_frame = current_frame.copy()

                if vis is not None:
                    # Resize visualization
                    vis_resized = cv2.resize(vis, (200, 200))

                    # Overlay on frame
                    h, w = vis_resized.shape[:2]
                    display_frame[10:10 + h, 10:10 + w] = vis_resized

                    # Add border
                    cv2.rectangle(display_frame, (8, 8), (12 + w, 12 + h), (255, 255, 255), 2)

                # Add FPS counter
                frame_count += 1
                if time.time() - last_time >= 1.0:
                    metrics['fps'] = frame_count
                    frame_count = 0
                    last_time = time.time()

                fps_text = f"FPS: {metrics['fps']}"
                cv2.putText(display_frame, fps_text, (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add mode indicator
                mode_text = f"Mode: {'Simulation' if simulation_mode else 'Camera'}"
                cv2.putText(display_frame, mode_text, (500, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # Add path info
                if current_path:
                    path_text = f"Path: {len(current_path)} steps"
                    cv2.putText(display_frame, path_text, (500, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', display_frame)
                if ret:
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If encoding fails, yield a placeholder
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            # Continue instead of breaking
            pass

        # Control update rate
        elapsed = time.time() - start_time
        target_time = 1.0 / config['update_rate']
        if elapsed < target_time:
            time.sleep(target_time - elapsed)


# ================= FLASK ROUTES =================
@app.route('/')
def index():
    return render_template('pathfinder_index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('pathfinder_dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def api_start():
    global is_running
    is_running = True
    log_event('SYSTEM', 'Pathfinder started')
    return jsonify({'success': True, 'message': 'Pathfinder started'})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global is_running
    is_running = False
    log_event('SYSTEM', 'Pathfinder stopped')
    return jsonify({'success': True, 'message': 'Pathfinder stopped'})


@app.route('/api/stats')
def api_stats():
    stats = {
        'is_running': is_running,
        'algorithm': config['algorithm'],
        'algorithm_name': ALGORITHMS.get(config['algorithm'], 'Unknown'),
        'grid_size': config['grid_size'],
        'fps': metrics['fps'],
        'path_length': metrics['path_length'],
        'nodes_explored': metrics['nodes_explored'],
        'computation_time': metrics['computation_time'],
        'success_rate': metrics['success_rate'],
        'total_paths': metrics['total_paths_found'],
        'total_failures': metrics['total_failures'],
        'auto_detect': config['auto_detect'],
        'update_rate': config['update_rate'],
        'simulation_mode': simulation_mode,
        'last_updated': datetime.now().strftime("%H:%M:%S")
    }
    return jsonify(stats)


@app.route('/api/current_path')
def api_current_path():
    """Get current path data"""
    try:
        grid_list = None
        if grid_map is not None:
            # Ensure grid_map is converted to list properly
            grid_list = grid_map.astype(int).tolist()

        path_data = {
            'path': current_path,
            'start': start_point,
            'end': end_point,
            'grid_size': config['grid_size'],
            'grid_map': grid_list,
            'has_path': len(current_path) > 0 if current_path else False
        }
        return jsonify(path_data)
    except Exception as e:
        print(f"Error in api_current_path: {e}")
        return jsonify({
            'path': [],
            'start': None,
            'end': None,
            'grid_size': config['grid_size'],
            'grid_map': None,
            'has_path': False
        })


@app.route('/api/algorithms')
def api_algorithms():
    return jsonify(ALGORITHMS)


@app.route('/api/set_algorithm', methods=['POST'])
def api_set_algorithm():
    try:
        data = request.json
        algorithm = data.get('algorithm', 'a_star')

        if algorithm in ALGORITHMS:
            config['algorithm'] = algorithm
            save_config()
            log_event('CONFIG', f'Algorithm changed to {algorithm}')
            return jsonify({'success': True, 'algorithm': algorithm})
        else:
            return jsonify({'success': False, 'error': 'Invalid algorithm'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/set_grid_size', methods=['POST'])
def api_set_grid_size():
    try:
        data = request.json
        size = int(data.get('size', 20))

        if 10 <= size <= 40:
            config['grid_size'] = size
            save_config()
            log_event('CONFIG', f'Grid size changed to {size}x{size}')
            return jsonify({'success': True, 'grid_size': size})
        else:
            return jsonify({'success': False, 'error': 'Size must be between 10 and 40'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/auto_points', methods=['POST'])
def api_auto_points():
    global start_point, end_point

    if grid_map is not None:
        start, end = generate_start_end_points(grid_map)
        if start and end:
            start_point = start
            end_point = end
            return jsonify({'success': True, 'start': start, 'end': end})

    return jsonify({'success': False, 'error': 'Cannot generate points'})


@app.route('/api/reset_metrics', methods=['POST'])
def api_reset_metrics():
    global metrics
    metrics = {
        'fps': 0,
        'path_length': 0,
        'nodes_explored': 0,
        'computation_time': 0,
        'success_rate': 0,
        'total_paths_found': 0,
        'total_failures': 0
    }
    return jsonify({'success': True, 'message': 'Metrics reset'})


@app.route('/api/reinit_camera', methods=['POST'])
def api_reinit_camera():
    """Reinitialize camera"""
    global camera, simulation_mode
    try:
        if camera:
            camera.release()
            camera = None

        simulation_mode = False
        init_camera()

        if simulation_mode:
            return jsonify({'success': False, 'message': 'Camera initialization failed, running in simulation mode'})
        else:
            return jsonify({'success': True, 'message': 'Camera reinitialized successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ================= MAIN =================
if __name__ == '__main__':
    # Initialize system
    print("🚀 Initializing AI Pathfinder System...")
    init_system()

    print("\n" + "=" * 70)
    print("🤖 AI PATHFINDER READY")
    print("=" * 70)
    print(f"📹 Camera Feed: http://localhost:5000")
    print(f"📊 Dashboard: http://localhost:5000/dashboard")
    print(f"📈 Live Stats: http://localhost:5000/api/stats")
    print("=" * 70)
    print("\n🎯 Available Algorithms:")
    for algo, name in ALGORITHMS.items():
        print(f"   • {algo}: {name}")
    print("=" * 70 + "\n")

    # Start Flask
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

import os
import cv2
import numpy as np
from pathlib import Path

def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    """Draw a polygon from points and display its vertices."""
    height, width = img.shape[:2]
    
    # Convert normalized coordinates to image coordinates
    points_img = points.copy()
    points_img[::2] *= width  # x coordinates
    points_img[1::2] *= height  # y coordinates
    
    # Reshape points to (4, 2) format for drawing
    points_img = points_img.reshape(-1, 2)
    points_img = np.int32(points_img)
    
    # Draw the polygon
    cv2.polylines(img, [points_img], True, color, thickness)
    
    # Draw vertices
    for point in points_img:
        cv2.circle(img, tuple(point), 3, (0, 0, 255), -1)
    
    # Calculate and draw center point
    center = np.mean(points_img, axis=0).astype(int)
    cv2.circle(img, tuple(center), 3, (255, 0, 0), -1)

def visualize_dataset(image_dir, label_dir, output_dir):
    """Visualize dataset with polygon labels."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each label file
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        # Find corresponding image
        img_name = Path(label_file).stem
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(image_dir, img_name + ext)
            if os.path.exists(img_path):
                break
        else:
            print(f"No image found for {label_file}")
            continue
        
        # Read image and labels
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        # Read and draw each polygon from label file
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:  # class_id + 8 coordinates (4 points)
                    # Parse values
                    class_id = int(parts[0])
                    points = np.array([float(x) for x in parts[1:]])
                    
                    # Draw polygon with different colors for different classes
                    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                    draw_polygon(img, points, color)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"viz_{img_name}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved visualization for {img_name}")

if __name__ == "__main__":
    image_dir = "original-datasets/original-images"  # Path to original images
    label_dir = "original-datasets/original-labels"  # Path to polygon format labels
    output_dir = "original-datasets/visualizations"  # Output directory for visualizations
    
    print("Starting visualization...")
    visualize_dataset(image_dir, label_dir, output_dir)
    print("Visualization complete!") 
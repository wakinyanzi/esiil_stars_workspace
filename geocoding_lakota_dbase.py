# geocode_map.py
import cv2
import numpy as np
import csv
from pathlib import Path

def detect_dots(OLCA_1979_Compressed, output_image=None):
    """Detect hand-drawn dots on map and return their pixel coordinates"""
    # Load image
    img = cv2.imread(OLCA_1979_Compressed)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {OLCA_1979_Compressed}")
    
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Threshold to isolate dots
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and get centroids
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 1000:  # Filter by dot size
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots.append((cx, cy))
    
    # Save visualization if requested
    if output_image:
        for (x, y) in dots:
            cv2.circle(img, (x, y), 5, (0, 0, 255), 2)
        cv2.imwrite(output_image, img)
    
    return np.array(dots)

def georeference(pixel_points, control_points):
    """
    Convert pixel coordinates to geographic coordinates
    :param pixel_points: Array of (x,y) pixel coordinates
    :param control_points: List of [(img_x, img_y, lon, lat), ...]
    :return: Array of (lon, lat) coordinates
    """
    src = np.array([(x, y) for x, y, _, _ in control_points], dtype=np.float32)
    dst = np.array([(lon, lat) for _, _, lon, lat in control_points], dtype=np.float32)
    
    # Calculate transformation matrix
    transform = cv2.findHomography(src, dst)[0]
    
    # Convert to homogeneous coordinates
    homogeneous = np.column_stack([pixel_points, np.ones(len(pixel_points))])
    
    # Apply transformation
    geo_coords = np.dot(transform, homogeneous.T).T
    geo_coords = geo_coords[:, :2] / geo_coords[:, 2:]  # Normalize
    
    return geo_coords

def save_to_csv(coordinates, output_file):
    """Save coordinates to CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['longitude', 'latitude'])
        writer.writerows(coordinates)

if __name__ == "__main__":
    # Configuration - EDIT THESE VALUES
    IMAGE_FILE = "historical_map.jpg"  # Your image in repo
    CONTROL_POINTS = [
        # (pixel_x, pixel_y, longitude, latitude)
        (100, 150, -101.2345, 43.1234),  # Top-left known point
        (2500, 300, -100.9876, 43.1567), # Top-right known point
        (800, 1800, -101.1234, 42.9876)  # Bottom known point
    ]
    OUTPUT_CSV = "coordinates.csv"
    VISUALIZATION = "detected_dots.jpg"  # Set None to disable
    
    try:
        # 1. Detect dots
        print("Detecting hand-drawn dots...")
        dots = detect_dots(IMAGE_FILE, VISUALIZATION)
        print(f"Found {len(dots)} dots")
        
        # 2. Georeference
        print("Calculating geographic coordinates...")
        geo_coords = georeference(dots, CONTROL_POINTS)
        
        # 3. Save results
        save_to_csv(geo_coords, OUTPUT_CSV)
        print(f"Saved coordinates to {OUTPUT_CSV}")
        
        if VISUALIZATION:
            print(f"Visualization saved to {VISUALIZATION}")
        
        print("\nNext steps:")
        print(f"1. git add {OUTPUT_CSV}")
        print(f"2. git add {VISUALIZATION}") if VISUALIZATION else None
        print("3. git commit -m 'Add geocoded coordinates'")
        print("4. git push")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Common fixes:")
        print("- Ensure image is in your repository root")
        print("- Verify control points are accurate")
        print("- Check dot contrast (try enhancing image if needed)")
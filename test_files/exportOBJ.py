import cv2
import numpy as np

img = cv2.imread("floorplan.jpeg", cv2.IMREAD_COLOR)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to obtain a binary image
_, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the biggest contour based on area
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

# Create a mask
mask = np.zeros_like(img)
cv2.fillPoly(mask, [biggest_contour], 255)

# Mark the outside of the house as black
img[mask == 0] = 1

# Approximate the contour to get the vertices
epsilon = 0.01 * cv2.arcLength(biggest_contour, True)  # You can adjust the epsilon value
approx_vertices = cv2.approxPolyDP(biggest_contour, epsilon, True)

# Assign a y coordinate to each vertex to create 3D vertices
fixed_y = 0.0  # You can set this to any desired value
vertices_3d = [(vertex[0][0], fixed_y, vertex[0][1]) for vertex in approx_vertices]

# Write the 3D vertices to an OBJ file
with open("contour.obj", "w") as file:
    file.write("# OBJ file\n")
    
    # Write vertices
    for vertex in vertices_3d:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
    # Write faces (assuming a single face from the contour vertices)
    file.write("f")
    for i in range(1, len(vertices_3d) + 1):
        file.write(f" {i}")
    file.write("\n")

print("OBJ file 'contour.obj' has been created.")

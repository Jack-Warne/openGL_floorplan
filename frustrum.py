import numpy as np
import glm
from camera import Camera

class Frustrum:
    @staticmethod
    def extract_frustum_planes(m):
        planes = []

        # Right plane
        planes.append((m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]))
        # Left plane
        planes.append((m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]))
        # Bottom plane
        planes.append((m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]))
        # Top plane
        planes.append((m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]))
        # Far plane
        planes.append((m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]))
        # Near plane
        planes.append((m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]))

        # Normalize planes
        for i in range(len(planes)):
            x, y, z, w = planes[i]
            length = glm.sqrt(x * x + y * y + z * z)
            planes[i] = (x / length, y / length, z / length, w / length)

        return planes
    def is_in_frustum(planes, bounding_box):
        """
        Check if a bounding box is inside the frustum.
        bounding_box: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        planes: list of frustum planes [(a, b, c, d), ...]
        """
        for plane in planes:
            normal = plane[:3]
            d = plane[3]

            # Find the point on the bounding box that is farthest along the normal
            far_point = [
                bounding_box[1][i] if normal[i] > 0 else bounding_box[0][i]
                for i in range(3)
            ]
            # Check if this point is outside the plane
            if np.dot(normal, far_point) + d < 0:
                return False  # Outside the frustum
        return True
    def is_inside_frustum(center, planes, radius):
        """
        Check if a sphere (center, radius) intersects the view frustum.
        """
        for plane in planes:
            distance = (
                plane[0] * center[0] + 
                plane[1] * center[1] + 
                plane[2] * center[2] + 
                plane[3]
            )
            if distance < -radius:
                return False
        return True
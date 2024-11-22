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
    def is_in_frustum(aabb_min, aabb_max, planes):
        """
        Check if an AABB (axis-aligned bounding box) is inside the view frustum.
        :param aabb_min: Minimum corner of the AABB (x_min, y_min, z_min).
        :param aabb_max: Maximum corner of the AABB (x_max, y_max, z_max).
        :param planes: Frustum planes, where each plane is (a, b, c, d) for ax + by + cz + d = 0.
        :return: True if the AABB intersects the frustum, False otherwise.
        """
        for plane in planes:
            a, b, c, d = plane
            # Test all 8 corners of the AABB against the current plane
            if (a * aabb_min[0] + b * aabb_min[1] + c * aabb_min[2] + d > 0 or
                a * aabb_max[0] + b * aabb_min[1] + c * aabb_min[2] + d > 0 or
                a * aabb_min[0] + b * aabb_max[1] + c * aabb_min[2] + d > 0 or
                a * aabb_max[0] + b * aabb_max[1] + c * aabb_min[2] + d > 0 or
                a * aabb_min[0] + b * aabb_min[1] + c * aabb_max[2] + d > 0 or
                a * aabb_max[0] + b * aabb_min[1] + c * aabb_max[2] + d > 0 or
                a * aabb_min[0] + b * aabb_max[1] + c * aabb_max[2] + d > 0 or
                a * aabb_max[0] + b * aabb_max[1] + c * aabb_max[2] + d > 0):
                continue  # At least one corner is in front of this plane
            return False  # All corners are behind this plane
        return True  # AABB intersects the frustum
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
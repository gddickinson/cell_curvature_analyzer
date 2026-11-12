"""Alternative edge-point sampling for CurvatureAnalyzer: angular, axis-based and x-axis distributions."""
import numpy as np


class AlternativeSamplingMixin:
    """CurvatureAnalyzer mixin: angular, axis-based and x-axis point sampling."""

    def sample_angular_points(self, contour, n_points=100):
        """
        Sample points around the contour at equal angular intervals.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly by angle
        """
        # Find the centroid of the cell (approximate center)
        center = np.mean(contour, axis=0)

        print(f"DEBUG: Cell centroid is at {center}")

        # Convert contour points to polar coordinates
        # Calculate the angle of each point relative to center
        polar_coords = []
        for point in contour:
            # Calculate vector from center to point
            vec = point - center
            # Calculate angle in radians
            angle = np.arctan2(vec[1], vec[0])
            # Calculate radius (distance from center)
            radius = np.linalg.norm(vec)
            polar_coords.append((angle, radius, point))

        # Sort by angle
        polar_coords.sort(key=lambda x: x[0])

        # Convert angles to cumulative form (unwrapped)
        unwrapped_coords = []
        prev_angle = polar_coords[0][0]
        cumulative_angle = prev_angle

        for angle, radius, point in polar_coords:
            # Handle angle wrap-around
            delta = angle - prev_angle
            if delta < -np.pi:
                delta += 2*np.pi
            elif delta > np.pi:
                delta -= 2*np.pi

            cumulative_angle += delta
            unwrapped_coords.append((cumulative_angle, radius, point))
            prev_angle = angle

        # Get min and max of cumulative angle
        min_angle = unwrapped_coords[0][0]
        max_angle = unwrapped_coords[-1][0]

        # Create target angles
        target_angles = np.linspace(min_angle, max_angle, n_points)

        # Find closest points to each target angle
        sampled_points = []
        for target in target_angles:
            closest_idx = min(range(len(unwrapped_coords)),
                              key=lambda i: abs(unwrapped_coords[i][0] - target))
            sampled_points.append(unwrapped_coords[closest_idx][2])

        # Check for duplicate points and adjust if necessary
        unique_points = []
        for i, point in enumerate(sampled_points):
            if i > 0 and np.array_equal(point, unique_points[-1]):
                # Find the next closest point to this target angle
                print(f"DEBUG: Found duplicate point at index {i}, finding alternative")
                target = target_angles[i]
                distances = [abs(coord[0] - target) for coord in unwrapped_coords]
                sorted_indices = np.argsort(distances)

                # Try the second closest point
                for idx in sorted_indices[1:5]:  # Try up to 5 alternative points
                    alt_point = unwrapped_coords[idx][2]
                    if not any(np.array_equal(alt_point, p) for p in unique_points):
                        unique_points.append(alt_point)
                        print(f"DEBUG: Found alternative point")
                        break
                else:
                    # If we can't find a unique point, slightly perturb the current one
                    perturbed = point + np.random.normal(0, 0.1, 2)
                    unique_points.append(perturbed)
                    print(f"DEBUG: Using perturbed point")
            else:
                unique_points.append(point)

        sampled_points = np.array(unique_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points using angular distribution")

        # Check for points that are too close to each other
        for i in range(len(sampled_points)):
            next_i = (i + 1) % len(sampled_points)  # Wrap around to the first point
            p1 = sampled_points[i]
            p2 = sampled_points[next_i]
            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-10:  # Very small distance
                print(f"DEBUG: Points {i} and {next_i} are very close: distance={dist}")

        return np.array(sampled_points)

    def sample_axis_based_points(self, contour, n_points=100):
        """
        Sample points around the contour with even spacing along the image axes.
        This helps reduce bunching in extrusions by ensuring points are distributed
        evenly in screen space rather than by contour distance or angle.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly by axis position
        """
        # Get the bounding box of the contour
        min_y, min_x = np.min(contour, axis=0)
        max_y, max_x = np.max(contour, axis=0)

        print(f"DEBUG: Contour bounding box: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")

        # We'll use a combination of x-axis and y-axis sampling
        # First, let's determine how many points to sample from each axis
        # We'll split them based on the aspect ratio of the bounding box
        width = max_x - min_x
        height = max_y - min_y

        # Calculate how many points to allocate to each axis based on perimeter
        perimeter = 2 * (width + height)
        x_portion = 2 * width / perimeter
        y_portion = 2 * height / perimeter

        n_x_points = int(n_points * x_portion)
        n_y_points = n_points - n_x_points

        print(f"DEBUG: Sampling {n_x_points} points from x-axis edges, {n_y_points} points from y-axis edges")

        # Create arrays of x and y coordinates for sampling
        x_values = []
        y_values = []

        # Sample points from the top and bottom edges
        if n_x_points > 0:
            x_step = width / (n_x_points / 2)

            # Top edge (constant y = min_y)
            for i in range(n_x_points // 2):
                x = min_x + i * x_step
                x_values.append(x)
                y_values.append(min_y)

            # Bottom edge (constant y = max_y)
            for i in range(n_x_points // 2):
                x = max_x - i * x_step
                x_values.append(x)
                y_values.append(max_y)

        # Sample points from the left and right edges
        if n_y_points > 0:
            y_step = height / (n_y_points / 2)

            # Left edge (constant x = min_x)
            for i in range(n_y_points // 2):
                y = min_y + i * y_step
                x_values.append(min_x)
                y_values.append(y)

            # Right edge (constant x = max_x)
            for i in range(n_y_points // 2):
                y = max_y - i * y_step
                x_values.append(max_x)
                y_values.append(y)

        # Create grid points from these coordinates
        grid_points = np.column_stack((y_values, x_values))

        # Find the closest contour point to each grid point
        sampled_points = []
        for grid_point in grid_points:
            # Calculate distances to all contour points
            distances = np.sqrt(np.sum((contour - grid_point)**2, axis=1))

            # Find the closest point
            closest_idx = np.argmin(distances)
            closest_point = contour[closest_idx]

            # Add to sampled points, checking for duplicates
            if not any(np.array_equal(closest_point, p) for p in sampled_points):
                sampled_points.append(closest_point)

        # If we don't have enough points (due to duplicates), add more
        if len(sampled_points) < n_points // 2:  # If we have far too few points
            print(f"DEBUG: Only found {len(sampled_points)} unique points, adding more")

            # Sort contour points by their perimeter distance
            distances = [0]
            for i in range(1, len(contour)):
                distances.append(distances[-1] + np.linalg.norm(contour[i] - contour[i-1]))

            total_length = distances[-1]

            # Sample additional points evenly by distance
            additional_points_needed = n_points - len(sampled_points)
            for i in range(additional_points_needed):
                target_dist = (i + 1) * total_length / (additional_points_needed + 1)

                # Find the closest point by distance
                idx = np.searchsorted(distances, target_dist)
                if idx >= len(contour):
                    idx = len(contour) - 1

                point = contour[idx]

                # Add if not already included
                if not any(np.array_equal(point, p) for p in sampled_points):
                    sampled_points.append(point)

        # Convert to numpy array
        sampled_points = np.array(sampled_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points using axis-based distribution")

        return sampled_points

    def sample_x_axis_points(self, contour, n_points=100):
        """
        Sample points along the contour with even spacing along the x-axis.
        This distributes points at regular x-intervals to prevent bunching
        in areas with membrane extrusions.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly along x-axis
        """
        # Sort contour points by x-coordinate
        x_sorted_indices = np.argsort(contour[:, 1])
        x_sorted_contour = contour[x_sorted_indices]

        # Get min and max x values
        min_x = x_sorted_contour[0, 1]
        max_x = x_sorted_contour[-1, 1]
        x_range = max_x - min_x

        print(f"DEBUG: X-axis range: [{min_x:.1f}, {max_x:.1f}]")

        # Calculate x-positions with even spacing
        x_positions = np.linspace(min_x, max_x, n_points)

        # For each x-position, find the closest point on the contour
        sampled_points = []
        for x_pos in x_positions:
            # Find the closest point by x-coordinate
            x_distances = np.abs(contour[:, 1] - x_pos)
            closest_idx = np.argmin(x_distances)
            closest_point = contour[closest_idx]

            # Add point to our sampled list
            sampled_points.append(closest_point)

        # Convert to numpy array
        sampled_points = np.array(sampled_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points based on x-axis positions")

        return sampled_points

from motion_code.motion_code import MotionCode
from motion_code.sparse_gp import sigmoid
import numpy as np
import heapq
import pdb

def find_k_furthest_points(arr, k):
    if k <= 0:
        return []
    # Step 1: Sort the array
    sorted_arr = sorted(arr)
    # Initialize the furthest points list with the largest point
    furthest_points = [sorted_arr[-1]]
    used_indices = {len(sorted_arr) - 1}
    # Step 2: Iteratively find the furthest points
    while len(furthest_points) < k:
        max_distance = -1
        next_point = None
        next_index = -1

        for i in range(len(sorted_arr)):
            if i in used_indices:
                continue

            # Calculate the minimum distance of this point to the current set of furthest points
            min_distance = min(abs(sorted_arr[i] - point) for point in furthest_points)

            # Update the furthest point if this distance is larger
            if min_distance > max_distance:
                max_distance = min_distance
                next_point = sorted_arr[i]
                next_index = i

        # Add the next furthest point to the list
        if next_point is not None:
            furthest_points.append(next_point)
            used_indices.add(next_index)

    return np.array(furthest_points)


def load_model(model_path = 'motion_code/saved_models/test_model'):
    model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)
    model.load(model_path)
    return model


def get_top_channels(num_motion=6, num_channels=224, top_k=24, dataset_name='jasper_ridge'):
    model = load_model(f'motion_code/saved_models/{dataset_name}')
    X_m, Z = model.X_m, model.Z
    X_m_ks = [sigmoid(X_m @ Z[k]) for k in range(num_motion)]
    X_m_ks = np.unique((np.array(X_m_ks) * num_channels).astype(int))
    X_m_ks = X_m_ks.flatten()
    if top_k > len(X_m_ks):
        top_k = len(X_m_ks)
    result = find_k_furthest_points(X_m_ks, top_k)
    result.sort()
    return result

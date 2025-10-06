import cv2
import numpy as np
from collections import Counter
try:
    from core.image_processor import get_best_contrast_channel, preprocess_with_clahe
except (ImportError, ModuleNotFoundError):
    pass


def find_line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    x1, y1 = line1_p1; x2, y2 = line1_p2; x3, y3 = line2_p1; x4, y4 = line2_p2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)

def refine_corners_subpixel_advanced(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    refined_corners_list = []
    for marker_corners_array in corners:
        refined1 = cv2.cornerSubPix(gray, marker_corners_array.copy(), (7, 7), (-1,-1), criteria)
        refined2 = cv2.cornerSubPix(gray, refined1.copy(), (5, 5), (-1, -1), criteria)
        refined_corners_list.append(refined2)
    return refined_corners_list

def find_box_pose(frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters):
    objp_marker15 = np.array([[0, marker_size_meters, 0], [marker_size_meters, marker_size_meters, 0], 
                               [marker_size_meters, 0, 0], [0, 0, 0]], dtype=np.float32)
    objp_marker5 = np.array([[-marker_size_meters, marker_size_meters, 0], [0, marker_size_meters, 0], 
                              [0, 0, 0], [-marker_size_meters, 0, 0]], dtype=np.float32)
    
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 35
    aruco_params.adaptiveThreshWinSizeStep = 5
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.polygonalApproxAccuracyRate = 0.04
    aruco_params.errorCorrectionRate = 0.7

    # --- 3-Stage Detection Logic ---
    ids = None; active_gray_image = None
    print("Attempt 1: Detecting on standard grayscale...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is not None:
        print(" -> Success with standard grayscale.")
        active_gray_image = gray
    if ids is None:
        print(" -> Failed. Attempt 2: Detecting on best contrast channel...")
        best_channel = get_best_contrast_channel(frame)
        corners, ids, _ = cv2.aruco.detectMarkers(best_channel, aruco_dict, parameters=aruco_params)
        if ids is not None:
            print(" -> Success with best contrast channel.")
            active_gray_image = best_channel
    if ids is None:
        print(" -> Failed. Attempt 3: Detecting with CLAHE...")
        clahe_image = preprocess_with_clahe(frame)
        corners, ids, _ = cv2.aruco.detectMarkers(clahe_image, aruco_dict, parameters=aruco_params)
        if ids is not None:
            print(" -> Success with CLAHE.")
            active_gray_image = clahe_image
    
    # --- Post-Detection Processing ---
    if ids is not None:
        pose_data = {}; corners_data = {}
        
        print("Applying advanced corner refinement...")
        refined_corners_list = refine_corners_subpixel_advanced(active_gray_image, corners)
        
        print("Applying line intersection refinement...")
        final_corners_list = []
        for marker_corners_array in refined_corners_list:
            p = marker_corners_array[0]
            tl, tr, br, bl = p[0], p[1], p[2], p[3]
            new_tl = find_line_intersection(tl, tr, tl, bl)
            new_tr = find_line_intersection(tr, tl, tr, br)
            new_br = find_line_intersection(br, tr, br, bl)
            new_bl = find_line_intersection(bl, br, bl, tl)
            if all(c is not None for c in [new_tl, new_tr, new_br, new_bl]):
                final_corners_list.append(np.array([[new_tl, new_tr, new_br, new_bl]], dtype=np.float32))
            else:
                final_corners_list.append(marker_corners_array)
        corners = tuple(final_corners_list)

        for i, marker_id_array in enumerate(ids):
            marker_id = marker_id_array[0]
            if marker_id == 5: obj_points = objp_marker5
            elif marker_id == 15: obj_points = objp_marker15
            else: continue
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                pose_data[marker_id] = {'rvec': rvec, 'tvec': tvec}
                corners_data[marker_id] = corners[i]

        if 5 in pose_data and 15 in pose_data:
            rvec5, tvec5 = pose_data[5]['rvec'], pose_data[5]['tvec']
            rvec15, tvec15 = pose_data[15]['rvec'], pose_data[15]['tvec']
            rot_mat5, _ = cv2.Rodrigues(rvec5)
            rot_mat15, _ = cv2.Rodrigues(rvec15)
            
            # ============================ IMPROVED ORIGIN CALCULATION ============================
            # Calculate axes
            length_axis = rot_mat15[:, 0].flatten()  # X-axis of marker 15 (points right)
            width_axis = -rot_mat5[:, 0].flatten()   # Negative X-axis of marker 5 (points into box)
            
            # Average the height axes for stability
            height_axis_5 = rot_mat5[:, 1].flatten()
            height_axis_15 = rot_mat15[:, 1].flatten()
            avg_height_axis = (height_axis_5 + height_axis_15) / 2.0
            height_axis = avg_height_axis / np.linalg.norm(avg_height_axis)
            
            # Calculate 3D positions of the relevant marker corners with border compensation
            border_width_meters = 0.01
            
            # Marker 5: bottom-right corner of QR pattern is at (0, 0, 0) in marker coords
            # We want the bottom-right corner of the WHITE BORDER
            # That's at (border_width, -border_width, 0) in marker 5's coordinate system
            marker5_corner_local = np.array([border_width_meters, -border_width_meters, 0])
            marker5_corner_3d = tvec5.flatten() + rot_mat5 @ marker5_corner_local
            
            # Marker 15: bottom-left corner of QR pattern is at (0, 0, 0) in marker coords
            # We want the bottom-left corner of the WHITE BORDER  
            # That's at (-border_width, -border_width, 0) in marker 15's coordinate system
            marker15_corner_local = np.array([-border_width_meters, -border_width_meters, 0])
            marker15_corner_3d = tvec15.flatten() + rot_mat15 @ marker15_corner_local
            
            # The box corner is the average of these two corner positions
            # This accounts for any slight misalignment between markers
            compensated_origin = (marker5_corner_3d + marker15_corner_3d) / 2.0
            
            # Round the origin to 0.1mm precision (0.0001m) for stability
            compensated_origin = np.round(compensated_origin, decimals=4)
            
            print(f"\nOrigin calculation:")
            print(f"  - Border width: {border_width_meters*100:.2f} cm")
            print(f"  - Marker 5 corner: {marker5_corner_3d}")
            print(f"  - Marker 15 corner: {marker15_corner_3d}")
            print(f"  - Final origin: {compensated_origin}")
            print(f"  - Distance between corners: {np.linalg.norm(marker5_corner_3d - marker15_corner_3d)*100:.2f} cm")
            # ========================================================================================

            results = {
                "common_origin": compensated_origin, 
                "length_axis": length_axis, 
                "width_axis": width_axis,
                "height_axis": height_axis, 
                "corners_data": corners_data, 
                "pose_data": pose_data
            }
            
            print("\nPose estimation successful!")
            return results
        else:
            print("Error: Could not find both markers 5 and 15 after processing.")
            return None
    
    print("Detection failed on all attempts. No markers found.")
    return None
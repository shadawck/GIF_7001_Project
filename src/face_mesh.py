import cv2
import numpy as np
import mediapipe as mp
import itertools

def extract_landmarks_points(face_mesh_results):
    keypoints = []
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        landmarks_face_points = face_landmarks.landmark

        for points_couple in landmarks_face_points :
            #print(points_couple)
            keypoints.append({
                     'X': points_couple.x,
                     'Y': points_couple.y,
                     'Z': points_couple.z,
                     })
    
    return keypoints

def extract_landmarks_points_as_numpy_ndarray(face_mesh_results):
    xyz = []

    for face_landmarks in face_mesh_results.multi_face_landmarks:
        landmarks_face_points = face_landmarks.landmark
        for points_couple in landmarks_face_points :
           xyz.append([points_couple.x,points_couple.y,points_couple.z])

    return np.array(xyz)


def video():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # GET webcam input and define a face detection geometry
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Detection for face geometries
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

                    # Detection for face contours
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # Detection for eyes
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def image():
    sample_img = cv2.imread('sample.jpg')
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                   min_detection_confidence=0.5)


    face_mesh_results = face_mesh_images.process(image)
    mp_drawing_styles = mp.solutions.drawing_styles
    # Get the list of indexes of the left and right eye.

    kp = extract_landmarks_points_as_numpy_ndarray(face_mesh_results)

    return kp

### Test KP extraction with Image
xyz = image()
print(xyz)


import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals()
normals = np.asarray(pcd.normals)
pcd.normals = o3d.utility.Vector3dVector(normals) # Normal flipping
pcd.orient_normals_to_align_with_direction()


o3d.io.write_point_cloud("face.ply", pcd)

pcd_load = o3d.io.read_point_cloud("face.ply")
xyz_load = np.asarray(pcd_load.points)
#o3d.visualization.draw_geometries([pcd_load])

### Use ball algo or other to construct mesh from point cloud

### BPA 
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist


### BPA Algo
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

### Poisson algo
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

o3d.io.write_triangle_mesh("bpa_mesh.ply", bpa_mesh)
o3d.io.write_triangle_mesh("poisson_mesh.ply", poisson_mesh)


bpa_load = o3d.io.read_point_cloud("bpa_mesh.ply")
o3d.visualization.draw_geometries([bpa_load], mesh_show_back_face=True)

poisson_load = o3d.io.read_point_cloud("poisson_mesh.ply")
o3d.visualization.draw_geometries([poisson_load], mesh_show_back_face=True)


pcd = o3d.io.read_point_cloud("bpa_mesh.ply")
o3d.visualization.draw_geometries([pcd], point_show_normal=True, mesh_show_back_face=True)

radii = [0.005, 0.01, 0.02, 0.04, 0.07, 0.1]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh], mesh_show_back_face=True)

pcd = o3d.io.read_point_cloud("face.ply")
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
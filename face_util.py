import face_recognition as fr
import numpy as np
import cv2

def compare_faces(file1, file2):
    """
    Compare two images and return True / False for matching.
    """
    # Load the jpg files into numpy arrays
    image1 = fr.load_image_file(file1)
    image2 = fr.load_image_file(file2)
    
    # Get the face encodings for each face in each image file
    # Assume there is only 1 face in each image, so get 1st face of an image.
    image1_encoding = fr.face_encodings(image1)[0]
    image2_encoding = fr.face_encodings(image2)[0]
    
    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = fr.compare_faces([image1_encoding], image2_encoding)    
    return results[0]



def detect_spoofing2(file):
    """
    Detect whether the image is a spoof or real.
    """
    # Check if the input is a file object or a file path
    if hasattr(file, 'read'):
        # Read the image data from the file object
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Read the image from the file path
        image = cv2.imread(file)

    if image is None:
        raise ValueError("Invalid image file")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Threshold to determine if the image is a spoof or real
    if laplacian_var < 100:
        return True  # Spoof
    else:
        return False  # Real

# Each face is tuple of (Name,sample image)
known_faces = [('Obama','sample_images/obama.jpg'),
               ('Peter','sample_images/peter.jpg'),
              ]

def face_rec(file):
    """
    Return name for a known face, otherwise return 'Unknown'.
    """
    for name, known_file in known_faces:
        if compare_faces(known_file,file):
            return name
    return 'Unknown' 
    
def find_facial_features(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all facial features in all the faces in the image
    face_landmarks_list = fr.face_landmarks(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_landmarks_list) != 1:
        return {}
    else:
        return face_landmarks_list[0]
        
def find_face_locations(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all face locations for the faces in the image
    face_locations = fr.face_locations(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_locations) != 1:
        return []
    else:
        return face_locations[0]        
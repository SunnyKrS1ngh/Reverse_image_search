import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False
model = tensorflow.keras.Sequential([
    resnet_model,
    GlobalMaxPooling2D()
])

# Function to recommend and display images
def recommend_images(file_path):
    # Clear previous recommendations
    for widget in recommended_images_frame.winfo_children():
        widget.destroy()
    
    # Load selected image
    img = Image.open(file_path)
    img = img.resize((224, 224))  # Resize to match model input size
    img_tk = ImageTk.PhotoImage(img)
    selected_image_label.configure(image=img_tk)
    selected_image_label.image = img_tk
    
    # Extract features from selected image
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized_result])
    
    # Display recommended images
    for i, file_index in enumerate(indices[0][1:6]):
        temp_img = cv2.imread(filenames[file_index])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        temp_img = Image.fromarray(temp_img)
        temp_img = ImageTk.PhotoImage(temp_img)
        label = tk.Label(recommended_images_frame, image=temp_img)
        label.image = temp_img
        label.grid(row=i, column=0, padx=10, pady=10)

# Function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        recommend_images(file_path)
    else:
        messagebox.showwarning("Warning", "No image selected.")

# Tkinter setup
window = tk.Tk()
window.title("Image Recommender")

selected_image_frame = tk.Frame(window)
selected_image_frame.pack(pady=10)

selected_image_label = tk.Label(selected_image_frame, text="Selected Image", font=("Helvetica", 14))
selected_image_label.pack()

recommended_images_frame = tk.Frame(window)
recommended_images_frame.pack(pady=10)

recommended_images_label = tk.Label(recommended_images_frame, text="Recommended Images", font=("Helvetica", 14))
recommended_images_label.pack()

select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack(pady=10)

window.mainloop()

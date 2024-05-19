import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

model_path = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\unet_skin_cancer_segmentation_model3.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)

def segment_and_crop(image_path, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Error loading image at {image_path}")
    
    print("Original image shape:", original_img.shape)
    
    img = cv2.resize(original_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    print("Input image shape:", img.shape)

    prediction = model.predict(img)
    print("Prediction shape:", prediction.shape)
    prediction = (prediction > 0.5).astype(np.uint8)[0, :, :, 0]

    prediction = cv2.resize(prediction, (original_img.shape[1], original_img.shape[0]))
    print("Resized prediction shape:", prediction.shape)

    kernel = np.ones((5, 5), np.uint8)
    prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
    prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found:", len(contours))

    segmented_img = original_img.copy()
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = original_img[y:y + h, x:x + w]
        cropped_images.append(cropped_img)
        cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    mask_overlay = cv2.addWeighted(original_img, 0.6, cv2.cvtColor(prediction * 255, cv2.COLOR_GRAY2BGR), 0.4, 0)

    return segmented_img, mask_overlay, cropped_images

def select_image():
    global panelA, panelB, cropped_panels, labels_to_hide, canvas_frame
    path = filedialog.askopenfilename()
    if len(path) > 0:
        try:
            for label in labels_to_hide:
                label.grid_remove()

            segmented_img, mask_overlay, cropped_imgs = segment_and_crop(path, model)
            
            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
            segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
            mask_overlay = Image.fromarray(mask_overlay)
            segmented_img = Image.fromarray(segmented_img)

            mask_overlay = mask_overlay.resize((500, 500), Image.LANCZOS)
            segmented_img = segmented_img.resize((500, 500), Image.LANCZOS)

            mask_overlay = ImageTk.PhotoImage(mask_overlay)
            segmented_img = ImageTk.PhotoImage(segmented_img)

            if panelA is None or panelB is None:
                panelA = tk.Label(image=mask_overlay)
                panelA.image = mask_overlay
                panelA.grid(row=1, column=0, padx=10, pady=10)

                panelB = tk.Label(image=segmented_img)
                panelB.image = segmented_img
                panelB.grid(row=1, column=1, padx=10, pady=10)

                tk.Label(root, text="Segmented Image with Mask Overlay", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=10, pady=10)
                tk.Label(root, text="Segmented Image with Bounding Boxes", font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=10, pady=10)
            else:
                panelA.configure(image=mask_overlay)
                panelB.configure(image=segmented_img)
                panelA.image = mask_overlay
                panelB.image = segmented_img

            if canvas_frame is not None:
                canvas_frame.destroy()

            canvas_frame = tk.Frame(root)
            canvas_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

            canvas = tk.Canvas(canvas_frame, width=window_width-40, height=300)
            scrollbar = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(
                    scrollregion=canvas.bbox("all")
                )
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(xscrollcommand=scrollbar.set)

            for i, cropped_img in enumerate(cropped_imgs):
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cropped_img = Image.fromarray(cropped_img)

                cropped_img = cropped_img.resize((250, 250), Image.LANCZOS)

                cropped_img = ImageTk.PhotoImage(cropped_img)

                label = tk.Label(scrollable_frame, image=cropped_img)
                label.image = cropped_img
                label.grid(row=0, column=i, padx=5, pady=5)
                cropped_panels.append(label)

                tk.Label(scrollable_frame, text=f"Cropped Image {i + 1}", font=("Helvetica", 12, "bold")).grid(row=1, column=i, padx=5, pady=5)

            canvas.pack(side="top", fill="both", expand=True)
            scrollbar.pack(side="bottom", fill="x")

            root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Skin Cancer Segmentation")

initial_width = 500
initial_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - initial_height / 2)
position_right = int(screen_width / 2 - initial_width / 2)

root.geometry(f"{initial_width}x{initial_height}+{position_right}+{position_top}")

label_title = tk.Label(root, text="Istanbul Commerce University", font=("Helvetica", 16, "bold"))
label_title.grid(row=0, column=0, columnspan=2, padx=100, pady=50)
label_arda = tk.Label(root, text="Arda Batuhan Aydın 200021986", font=("Helvetica", 12))
label_arda.grid(row=1, column=0, columnspan=2, padx=100, pady=10)
label_umut = tk.Label(root, text="Umut Çakır 200022253", font=("Helvetica", 12))
label_umut.grid(row=2, column=0, columnspan=2, padx=100, pady=10)
label_arda2 = tk.Label(root, text="Arda Özhan 200022213", font=("Helvetica", 12))
label_arda2.grid(row=3, column=0, columnspan=2, padx=100, pady=10)

labels_to_hide = [label_title, label_arda, label_umut, label_arda2]

panelA = None
panelB = None

cropped_panels = []
canvas_frame = None

btn = tk.Button(root, text="Select an image", command=select_image)
btn.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

window_width = 1050
window_height = 1000
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

root.mainloop()
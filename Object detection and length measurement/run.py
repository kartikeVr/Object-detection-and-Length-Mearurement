import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from dectector import Detector

class App:
    def __init__(self, root):
        self.detector = Detector()
        self.root = root
        self.root.title("Object Detection GUI")
        self.root.geometry("900x700")

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 12))

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.upload_frame = ttk.Frame(self.main_frame, padding="10")
        self.upload_frame.pack(fill=tk.X)

        self.upload_image_button = ttk.Button(self.upload_frame, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack(side=tk.LEFT)

        self.upload_video_button = ttk.Button(self.upload_frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.pack(side=tk.LEFT, padx=(10, 0))

        self.detect_button = ttk.Button(self.upload_frame, text="Detect Objects", command=self.detect_objects, state=tk.DISABLED)
        self.detect_button.pack(side=tk.LEFT, padx=(10, 0))

        self.status_label = ttk.Label(self.upload_frame, text="No image/video uploaded", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

        self.image_frame = ttk.Frame(self.main_frame, padding="10")
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.load_model()

    def load_model(self):
        self.status_label.config(text="Loading model...", foreground="blue")
        self.root.update_idletasks()

        modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        classFile = "coco.names"

        try:
            self.detector.readClasses(classFile)
            self.detector.download(modelUrl)
            self.detector.loadModel()
            self.status_label.config(text="Model loaded successfully!", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}", foreground="red")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not self.image_path:
            return
        image = Image.open(self.image_path)
        image.thumbnail((800, 600))
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk)
        self.detect_button.config(state=tk.NORMAL)
        self.status_label.config(text="Image uploaded successfully.", foreground="green")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not self.video_path:
            return
        self.detect_button.config(state=tk.NORMAL)
        self.status_label.config(text="Video uploaded successfully.", foreground="green")

    def detect_objects(self):
        if hasattr(self, 'image_path'):
            self.detect_image()
        elif hasattr(self, 'video_path'):
            self.detect_video()
        else:
            messagebox.showerror("Error", "No image or video uploaded.")

    def detect_image(self):
        self.status_label.config(text="Detecting objects in image...", foreground="blue")
        self.root.update_idletasks()

        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Unable to read image.")

            result_image = self.detector.createIdentifier(image)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_image_pil = Image.fromarray(result_image)
            result_image_pil.thumbnail((800, 600))
            self.result_image_tk = ImageTk.PhotoImage(result_image_pil)
            self.image_label.config(image=self.result_image_tk)
            self.status_label.config(text="Detection completed.", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Error during detection: {str(e)}")
            self.status_label.config(text="Detection failed.", foreground="red")

def detect_video(self):
    self.status_label.config(text="Detecting objects in video...", foreground="blue")
    self.root.update_idletasks()

    cap = cv2.VideoCapture(self.video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return

  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])

    if not output_path:
        self.status_label.config(text="Video saving canceled.", foreground="red")
        return

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result_frame = self.detector.createIdentifier(frame)
            out.write(result_frame)  
            cv2.imshow("Video Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="Video detection completed and saved.", foreground="green")
    except Exception as e:
        out.release()  
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showerror("Error", f"Error during video detection: {str(e)}")
        self.status_label.config(text="Detection failed.", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

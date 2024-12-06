import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from dectector import Detector

class App:
    def __init__(self, root):
        self.scale = 3
        self.wP = 200 * self.scale
        self.hP = 287 * self.scale
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

        self.live_stream_button = ttk.Button(self.upload_frame, text="Live Stream", command=self.live_stream)
        self.live_stream_button.pack(side=tk.LEFT, padx=(10, 0))

        self.measure_length_button = ttk.Button(self.upload_frame, text="live measure", command=self.measure_length)
        self.measure_length_button.pack(side=tk.LEFT, padx=(10, 0))

        self.detect_button = ttk.Button(self.upload_frame, text="Detect Objects", command=self.detect_objects, state=tk.DISABLED)
        self.detect_button.pack(side=tk.LEFT, padx=(10, 0))

        self.status_label = ttk.Label(self.upload_frame, text="No image/video uploaded", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

        self.image_frame = ttk.Frame(self.main_frame, padding="10")
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.load_model()

    def getContours(self,img, cThr=[100, 100], showcanny=False, minArea=1000, filter=0, draw=False):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgblur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        cannyimg = cv2.Canny(imgblur, cThr[0], cThr[1])
        if showcanny:
            cv2.imshow("canny", cannyimg)
        kernel = np.ones((5, 5))
        imgdial = cv2.dilate(cannyimg, kernel=kernel, iterations=3)
        imgthre = cv2.erode(imgdial, kernel, iterations=2)

        contour, hierarchy = cv2.findContours(imgthre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finalContours = []

        for j, i in enumerate(contour):
            area = cv2.contourArea(i)
            if area > minArea:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                bbox = cv2.boundingRect(approx)
                if filter > 0:
                    if len(approx) == filter:
                        finalContours.append([len(approx), area, approx, bbox, i])
                else:
                    finalContours.append([len(approx), area, approx, bbox, i])
        finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
        if draw:
            for con in finalContours:
                cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
        return img, finalContours

    def reorder_points(self,pts):
        if len(pts) > 4:
            peri = cv2.arcLength(pts, True)
            approx = cv2.approxPolyDP(pts, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx
            else:
                return None
        if len(pts) != 4:
            raise ValueError(f"Expected 4 points to reorder, got {len(pts)}. Points: {pts}")

        pts = pts[np.argsort(pts[:, 0]), :]
        left_most = pts[:2, :]
        right_most = pts[2:, :]

        left_most = left_most[np.argsort(left_most[:, 1]), :]
        top_left, bottom_left = left_most

        right_most = right_most[np.argsort(right_most[:, 1]), :]
        top_right, bottom_right = right_most

        return np.array([top_left, top_right, bottom_left, bottom_right], dtype='float32')

    def warp_image(self,img, points, w, h, pad=10):
        points = self.reorder_points(points)
        if points is None:
            return img
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, matrix, (w, h))
        warped = warped[pad:warped.shape[0] - pad, pad:warped.shape[1] - pad]
        return warped

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

    def live_stream(self):
        self.status_label.config(text="Starting live stream...", foreground="blue")
        self.root.update_idletasks()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.status_label.config(text="Failed to grab frame.", foreground="red")
                    break
                result_frame = self.detector.createIdentifier(frame)
                cv2.imshow("Live Stream", result_frame  )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            self.status_label.config(text="Live stream ended.", foreground="green")

        except Exception as e:
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showerror("Error", f"Error during live stream: {str(e)}")
            self.status_label.config(text="Live stream failed.", foreground="red")


    def measure_length(self):
        self.status_label.config(text="Starting live stream...", foreground="blue")
        self.root.update_idletasks()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.status_label.config(text="Failed to grab frame.", foreground="red")
                    break

                frame = cv2.resize(frame, (900, 500))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 1)
                edges = cv2.Canny(blurred, 50, 150)
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=1)
                eroded = cv2.erode(dilated, kernel, iterations=1)

                contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                if contours:
                    largest_contour = contours[0]
                    peri = cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

                    points = approx.reshape(len(approx), 2)

                    warped = self.warp_image(frame, points, self.hP, self.wP)

                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
                    edges = cv2.Canny(blurred, 100, 100)
                    kernel = np.ones((5, 5))
                    imgDial = cv2.dilate(edges, kernel, iterations=3)
                    imgThr = cv2.erode(imgDial, kernel, iterations=2)

                    contours, _ = cv2.findContours(imgThr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img2, conts2 = self.getContours(warped, minArea=2000, filter=0, cThr=[50, 50],
                                               draw=False)

                    for obj in conts2:
                        rect = cv2.minAreaRect(obj[4])
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        x, y, w, h = obj[3]
                        cv2.drawContours(img2, [box], 0, (0, 255, 0), 2)

                        (width, height) = rect[1]
                        width_cm = round(width / self.scale / 10, 1)
                        height_cm = round(height / self.scale / 10, 1)

                        cv2.putText(img2, f"W: {width_cm}cm", (int(rect[0][0] - 50), int(rect[0][1] - 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        cv2.putText(img2, f"H: {height_cm}cm", (int(rect[0][0] - 50), int(rect[0][1] + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        ymin, xmin, ymax, xmax = obj[3]
                        xmin, xmax, ymin, ymax = (xmin * width, xmax * width, ymin * height, ymax * height)
                        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                        cv2.rectangle(img2, (xmin, ymin), (xmax, ymax), color=(0, 210, 0), thickness=1)
                        cv2.putText(img2, f"W: {width_cm}cm \n H: {height_cm}cm ", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1,
                                    (0, 210, 0), 2)

                        lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                        cv2.line(img2, (xmin, ymin), (xmin + lineWidth, ymin), (0, 210, 0), thickness=5)
                        cv2.line(img2, (xmin, ymin), (xmin, ymin + lineWidth), (0, 210, 0), thickness=5)

                        cv2.line(img2, (xmax, ymin), (xmax - lineWidth, ymin), (0, 210, 0), thickness=5)
                        cv2.line(img2, (xmax, ymin), (xmax, ymin + lineWidth), (0, 210, 0), thickness=5)

                result_frame = self.detector.createIdentifier(img2)
                cv2.imshow("Live Stream", img2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            self.status_label.config(text="Live stream ended.", foreground="green")
        except Exception as e:
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showerror("Error", f"Error during live stream: {str(e)}")
            self.status_label.config(text="Live stream failed.", foreground="red")

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

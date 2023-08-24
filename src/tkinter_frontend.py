import tkinter
from tkinter.ttk import *
from tkinter.filedialog import askdirectory

import cv2
from os.path import join, basename
from util.helpers import get_file_paths_with_extensions
from preprocessing.averaging import apply_profile_to_image, calculate_profile
from configuration import global_configuration

import threading

class Frontend(tkinter.Tk):
    def __init__(self, screenName: str | None = None, baseName: str | None = None, className: str = "NerveStitcher", useTk: bool = True, sync: bool = False, use: str | None = None) -> None:
        super().__init__(screenName, baseName, className, useTk, sync, use) 
        self.title("NerveStitcher")
        self.geometry("600x300")
        self.notebook = Notebook(self)
        self.notebook.pack(pady=0,expand=True,fill="both")

        self.preprocessing_input_directory = tkinter.StringVar()
        self.preprocessing_output_directory = tkinter.StringVar()
        self.stitching_input_directory = tkinter.StringVar()
        self.stitching_output_directory = tkinter.StringVar()

        self.preprocessing_tab()
        self.stitching_tab()
        self.progressbar = Progressbar(self, mode="indeterminate")
        self.progressbar.pack(side="bottom",fill="x")

    def preprocessing_tab(self):
        frame = Frame(self.notebook)
        self.notebook.add(frame,text="Preprocessing")
        frame.columnconfigure(0,weight=0)
        frame.columnconfigure(1,weight=1)
        frame.columnconfigure(2,weight=0)

        Label(frame,text="Input Directory:").grid(row=0,column=0,padx=5)
        Entry(frame,textvariable=self.preprocessing_input_directory).grid(row=0,column=1,sticky="ew")
        Button(frame,text="Open Folder",command=lambda: self.preprocessing_input_directory.set(askdirectory())).grid(row=0,column=2)

        Label(frame,text="Output Directory:").grid(row=1,column=0,padx=5)
        Entry(frame,textvariable=self.preprocessing_output_directory).grid(row=1,column=1,sticky="ew")
        Button(frame,text="Open Folder",command=lambda: self.preprocessing_output_directory.set(askdirectory())).grid(row=1,column=2)

        Button(frame,text="Process",command=self.preprocess).grid(row=2,columnspan=3,sticky="ew")
        self.notebook.pack(expand=True,fill="both")

    def stitching_tab(self):
        frame = Frame(self.notebook)
        self.notebook.add(frame,text="Stitching")


    def preprocess(self):
        
        def worker():
            image_paths = get_file_paths_with_extensions(self.preprocessing_input_directory.get(),global_configuration.supported_file_extensions)
            images = [cv2.imread(image_path) for image_path in image_paths]
            profile = calculate_profile(images)
            images = [apply_profile_to_image(image,profile) for image in images]
            image_paths = [join(self.preprocessing_output_directory.get(),basename(image_path)) for image_path in image_paths]
            for image, path in zip(images,image_paths):
                cv2.imwrite(path,image)
            self.progressbar.stop()
        
        self.progressbar.start(5)
        threading.Thread(target=worker).start()

frontend = Frontend()
frontend.mainloop()
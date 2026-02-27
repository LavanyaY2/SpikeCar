import argparse
import tkinter as tk
from tkinter import ttk
import timeit

import numpy as np
from PIL import Image, ImageTk
from PIL import ImageOps

import dvs

parser = argparse.ArgumentParser('Samsung DVS Viewer')
parser.add_argument('--source', metavar='FILE', type=str, default=None,
                    help='event file to view')
parser.add_argument('--timesurface_tau', default=0.01, type=float, 
                    help='filter time for event image view (default=0.01)')
parser.add_argument('--timesurface_gain', default=1.0, type=float, 
                    help='magnitude of event image view (default=1)')


args = parser.parse_args()

lib = dvs.Library(width=640, height=480)



class MainView:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Samsung DVS Viewer')

        self.root.geometry('1200x600')
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        self.pane1 = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.pane1.grid(row=0, column=0, sticky='news')
        self.frame1 = ttk.Frame(self.pane1, width=600, height=600)

        self.statusbar = ttk.Label(self.root, text='')
        self.statusbar.grid(row=1, column=0, sticky='news')

        self.has_image = False
        self.bind_keys(self.root)
        

        self.app_image = Timesurface(self, self.frame1)

    def toggle_image(self, event=None):
        if not self.has_image:
            if len(self.pane1.panes()) == 0:
                self.pane1.add(self.frame1, weight=1)
            else:
                self.pane1.insert(0, self.frame1, weight=1)
            self.has_image = True
        else:
            self.pane1.forget(0)
            self.has_image = False


    def bind_keys(self, window):
        window.bind('<Up>', self.image_gain_up)
        window.bind('<Down>', self.image_gain_down)
        window.bind('<Escape>', self.quit)
        

    def image_gain_up(self, event=None):
        args.timesurface_gain *= 1.2
    def image_gain_down(self, event=None):
        args.timesurface_gain /= 1.2


    def quit(self, event):
        self.root.destroy()


    def set_source(self):
        lib.lib.initialize_camera()
        lib.lib.start_camera()
        self.data = None
        self.root.after(10, self.update_data)

    def update_data(self):
        counts = []
        for i in range(3):
            counts.append(lib.lib.update_camera())
        self.root.after(10, self.update_data)





class Timesurface:
    def __init__(self, app, frame):
        self.app = app

        self.image = ImageTk.PhotoImage(image=self.get_image())
        self.label = ttk.Label(frame, image=self.image, padding=0)
        self.label.grid(column=0, rowspan=1, row=0, sticky='news')
        frame.bind('<Configure>', self.resize_image)
        app.root.update()
        app.root.after(50, self.update)

    def update(self):
        if not self.app.has_image:
            if lib.timesurface_active:
                lib.stop_timesurface()
        else:
            if not lib.timesurface_active:
                lib.init_timesurface(timesurface_tau=args.timesurface_tau)

            img = self.get_image()
            img = img.resize((self.image.width(), self.image.height()))
            self.image = ImageTk.PhotoImage(image=img)
            self.label.config(image=self.image)
        self.app.root.after(50, self.update)

    def resize_image(self, event):
        img = self.get_image()
        img = img.resize((event.width, event.height))
        self.image = ImageTk.PhotoImage(image=img)
        self.label.config(image=self.image)

    def get_image(self):
        if not lib.timesurface_active:
            img = np.zeros((640,480)).astype(np.uint8) + 128
        else:
            img = np.clip(lib.timesurface*args.timesurface_gain+128,0,255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        return img



main = MainView()
main.set_source()
main.toggle_image()
main.root.mainloop()



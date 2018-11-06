import os
import pandas as pd

####################################################################
# This script parses the .csv of HIT results, visualizes the results
# provides a GUI to accept or reject the results and creates a result
# .csv to upload on Mech Turk
####################################################################


import os
import pandas as pd
import sys
from PIL import Image, ImageTk, ImageDraw
import tkinter
import json
import shutil
from tkinter import Tk,Button,HORIZONTAL
from tkinter.ttk import Progressbar

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

####################################################################
# This script parses the .csv of HIT results, visualizes the results
# provides a GUI to accept or reject the results and creates a result
# .csv to upload on Mech Turk
####################################################################


# Variables#
# path of the HIT results

hit_path = "../resources/Batch_3408616_batch_results.csv"
data_path = "../resources/imgs"
# edited - Sahil
redo = "../resources/redo/"
rejected = "../resources/rejected/"

fields = [""]


# get image list
def get_images():
    image_list = []
    for im_path in os.listdir(data_path):
        img = os.path.join(data_path, im_path)
        image_list.append(img)
    return image_list


def get_images_annotations(df):
    hit_df_filtered = df[["HITId", "Input.image_url", "Answer.annotation_data"]]
    image_annotation_list = []
    for index, row in hit_df_filtered.iterrows():
        # print(row["Input.image_url"], row["Answer.annotation_data"])
        worker_answer = json.loads(row["Answer.annotation_data"])
        img = os.path.join(data_path, row["Input.image_url"])
        image_annotation_list.append((index, img, worker_answer))
    return image_annotation_list[::-1]


def tk_image(path, w, h, bbox):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    # draw.rectangle([left, top, right, bottom])
    for box in bbox:
        top = float(box['top'])
        left = float(box['left'])
        bottom = float(box['top']) + float(box['height'])
        right = float(box['left']) + float(box['width'])
        draw.line([(left, top), (right, top), (right, bottom), \
               (left, bottom), (left, top)], width=3)
    img = img.resize((w, h))
    storeobj = ImageTk.PhotoImage(img)
    return storeobj


# Creating Canvas Widget
class PictureWindow():
    def __init__(self, master, **kwargs):
        self.annotation_df = pd.read_csv(hit_path)
        self.current_hit = ""
        self.imagelist = get_images_annotations(self.annotation_df)
        print(self.annotation_df.count(axis='rows')[0])
        self.imagelist_p = []
        self.current_image = ''
        self.result_Dictionary = {}
        self.master = master
        for key, val in kwargs.items():
            if key == "width":
                self.w = int(val)
            elif key == "height":
                self.h = int(val)
        self.result = tkinter.IntVar()
        self.setup_gui()

    def on_close(self):
        print("in closing")
        self.annotation_df.to_csv("modified.csv")
        return

    def reverse(tuple):
        new_tuple = ()
        for x in reversed(tuple):
            new_tuple = new_tuple + (k,)
        return new_tuple

    def show_image(self, pop_tuple):
        path = pop_tuple[1]
        bbox = pop_tuple[2]
        
        img = tk_image(path, self.w, self.h, bbox)
        self.img_canvas.delete(self.img_canvas.find_withtag("bacl"))
        self.allready = self.img_canvas.create_image(self.w / 2, self.h / 2, image=img, anchor='center', tag="bacl")

        self.image = img
        self.current_hit = pop_tuple[0]
        #print(self.img_canvas.find_withtag("bacl"))
        #print(self.current_hit)
        self.master.title("Image Viewer ({})".format(path))
        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            #previous_result = self.result.pop()

            self.show_image(pop)
            self.imagelist.append(pop)
            # edited - Sahil
            if self.result_Dictionary.get(self.current_hit) is not None:
                self.result.set(self.result_Dictionary.get(self.current_hit))
            else:
                self.result.set(0)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()
            self.show_image(pop)
            self.imagelist_p.append(pop)
            # edited - Sahil
            if self.result_Dictionary.get(self.current_hit) is not None:
                self.result.set(self.result_Dictionary.get(self.current_hit))
            else:
                self.result.set(0)
        except EOFError as e:
            pass
        return

    def update_approval(self):
        selection = self.result.get()
        print(self.annotation_df.loc[self.current_hit, "Input.image_url"])
        if selection == 1:
            self.annotation_df.loc[self.current_hit, "Approve"] = "X"
            print(self.annotation_df.loc[self.current_hit, "Approve"])
            # edited - Sahil
            self.result_Dictionary.update({self.current_hit : 1})
            #print(self.result_Dictionary.get(self.current_hit))
        elif selection == 2:
            self.annotation_df.loc[self.current_hit, "Reject"] = "incorrect annotation"
            # edited - Sahil
            self.result_Dictionary.update({self.current_hit : 2})
            self.current_image = self.annotation_df.loc[self.current_hit,"Input.image_url"]
            shutil.copy2(data_path+"/"+self.current_image,rejected+self.current_image)
        return

    def setup_gui(self):
        self.img_canvas = tkinter.Canvas(self.master, width=self.w, height=self.h)
        self.create_buttons(self.img_canvas)
        self.img_canvas.pack()
        self.control_frame = tkinter.Frame(self.master)
        tkinter.Radiobutton(self.control_frame, text="Accept", padx=10, \
                            variable=self.result, value= 1, \
                            command=self.update_approval).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(self.control_frame, text="Reject", padx=10, \
                            variable=self.result, value= 2, \
                            command=self.update_approval).pack(side=tkinter.RIGHT)
        self.control_frame.pack(side=tkinter.BOTTOM)
        # self.window_settings()
        return

    def create_buttons(self, c):
        tkinter.Button(c, text=" > ", command=self.next_image).place(x=(self.w / 1.1), y=(self.h / 2))
        tkinter.Button(c, text=" < ", command=self.previous_image).place(x=20, y=(self.h / 2))
        c['bg'] = "white"
        return


# Main Function
def main():
    # Creating Window
    root = tkinter.Tk(className=" Image Viewer")
    # Creating the main Widget
    app = PictureWindow(root, width=1280, height=1024)
    # Not Resizable
    root.resizable(width=1280, height=1024)

    # Window Mainloop
    root.mainloop()
    app.on_close()
    return


# Main Function Trigger
if __name__ == '__main__':
    main()








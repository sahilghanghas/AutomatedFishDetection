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

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

####################################################################
# This script parses the .csv of HIT results, visualizes the results
# provides a GUI to accept or reject the results and creates a result
# .csv to upload on Mech Turk
####################################################################


# Variables#
# path of the HIT results
hit_path = "resources/Batch_3400145_batch_results.csv"
data_path = "resources/imgs"
fields = [""]



# get image list
def get_images():
    image_list = []
    for im_path in os.listdir(data_path):
        img = os.path.join(data_path, im_path)
        image_list.append(img)
    return image_list


def get_images_annotations(df):
    hit_df_filtered = df[["HITId", "Input.image_url", \
                          "Answer.annotation_data"]]
    image_annotation_list = []
    for index, row in hit_df_filtered.iterrows():
        # print(row["Input.image_url"], row["Answer.annotation_data"])
        worker_answer = json.loads(row["Answer.annotation_data"])
        img = os.path.join(data_path, row["Input.image_url"])
        image_annotation_list.append((index, img, worker_answer))
    return image_annotation_list


def tk_image(path, w, h, left, top, right, bottom):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    # draw.rectangle([left, top, right, bottom])
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
        self.imagelist_p = []
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

    def show_image(self, pop_tuple):
        path = pop_tuple[1]
        bbox = pop_tuple[2]
        top = float(bbox[0]['top'])
        left = float(bbox[0]['left'])
        bottom = float(bbox[0]['top']) + float(bbox[0]['height'])
        right = float(bbox[0]['left']) + float(bbox[0]['width'])
        img = tk_image(path, self.w, self.h, left, top, right, bottom)
        self.img_canvas.delete(self.img_canvas.find_withtag("bacl"))
        self.allready = self.img_canvas.create_image(self.w / 2, self.h / 2, image=img, anchor='center', tag="bacl")

        self.image = img
        self.current_hit = pop_tuple[0]
        print(self.img_canvas.find_withtag("bacl"))
        print(self.current_hit)
        self.master.title("Image Viewer ({})".format(path))
        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            self.show_image(pop)
            self.imagelist.append(pop)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()

            self.show_image(pop)
            self.imagelist_p.append(pop)
        except EOFError as e:
            pass
        return

    def update_approval(self):
        selection = self.result.get()
        if selection == 1:
            self.annotation_df.loc[self.current_hit, "Approve"] = "X"
            print(self.annotation_df.loc[self.current_hit, "Approve"])
        elif selection == 2:
            self.annotation_df.loc[self.current_hit, "Reject"] = "incorrect annotation"

        return

    def setup_gui(self):
        self.img_canvas = tkinter.Canvas(self.master, width=self.w, height=self.h)
        self.create_buttons(self.img_canvas)
        self.img_canvas.pack()

        self.control_frame = tkinter.Frame(self.master)
        tkinter.Radiobutton(self.control_frame, text="Accept", padx=10, \
                            variable=self.result, value=1, \
                            command=self.update_approval).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(self.control_frame, text="Reject", padx=10, \
                            variable=self.result, value=2, \
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
    app = PictureWindow(root, width=600, height=450)
    # Not Resizable
    root.resizable(width=0, height=0)

    # Window Mainloop
    root.mainloop()
    app.on_close()
    return


# Main Function Trigger
if __name__ == '__main__':
    main()








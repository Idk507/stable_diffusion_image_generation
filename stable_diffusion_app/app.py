import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline




print(torch.__version__)
print(torch.cuda.is_available())


app = ctk.CTk()

app.title("Stable Generation App With Stable Diffusion Model - Dhanushkumar.R")
ctk.set_appearance_mode("dark")


entry_frame = ctk.CTkFrame(app)
entry_frame.place(x=10, y=10)

prompt = ctk.CTkEntry(entry_frame, height=40, width=512, text_color="black", fg_color="white")
prompt.pack()

label_frame = ctk.CTkFrame(app)
label_frame.place(x=10, y=110)

lmain = ctk.CTkLabel(label_frame, height=512, width=512)
lmain.pack()

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

'''def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 
    '''
    
def generate():
    with autocast(device):
        image = pipe(str(prompt.get())).images[0]

    image.save('generatedimage.png')
    img = ctk.CTkImage(image, size=(512, 512))
    lmain.configure(image=img)

trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text='Generate')
trigger.place(x=206, y=60)

app.mainloop()
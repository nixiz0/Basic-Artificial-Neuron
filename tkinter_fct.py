from PIL import Image, ImageTk


# Design for the simpledialog (user input)
def app_theme(root):
    root.tk_setPalette(background='#333333', foreground='white')
    
def center_window(window):
    window_width = window.winfo_reqwidth()
    window_height = window.winfo_reqheight()
    position_right = int(window.winfo_screenwidth() / 2 - window_width / 2)
    position_down = int(window.winfo_screenheight() / 2 - window_height / 0.65)
    window.geometry(f'+{position_right}+{position_down}')
    
# Function to load and resize image
def load_and_resize_image(image_path, width, height):
    img = Image.open(image_path)
    img = img.resize((width, height), resample=Image.BILINEAR)
    return ImageTk.PhotoImage(img)
import os
from tkinter import *
from tkinter import filedialog

from PIL import  Image, ImageTk
import chan_vese as chan_vese

def list_image_names(directory="images"):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_names = [f for f in os.listdir(directory) if f.endswith(valid_extensions)]
    return image_names

def on_image_click(image_name,label, all_labels, directory="images"):
    image_path = os.path.join(directory, image_name)
    chan_vese.setImagePath(image_path)

    button_segmentation.config(text=f"Chan-Vese Segmentation: {image_name}")
    for lbl in all_labels:
        lbl.config(bg="#D9D9D9", fg="#6c6b6b")

    label.config(bg="#D9D9D9", fg="black")
    load_image(image_path)

def load_image(image_path):
    global img_label
    img = Image.open(image_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)

    if 'img_label' in globals():
        img_label.config(image=img)
        img_label.image = img
    else:
        img_label = Label(root, image=img)
        img_label.image = img
        img_label.grid(row=0, column=3)

def load_image_from_PC():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        load_image(file_path)
        chan_vese.setImagePath(file_path)
        button_segmentation.config(text=f"Chan-Vese Segmentation: {os.path.basename(file_path)}")
        for lbl in labels:
            lbl.config(bg="#D9D9D9", fg="black")

def setParams(new_mu, new_nu, new_eta, new_time_step,
              new_epsilon,new_expPhs):
   
    chan_vese.setParams(new_mu, new_nu, new_eta, new_time_step, new_epsilon)
    new_mu, new_nu, new_eta, new_time_step, new_epsilon  =chan_vese.getParams()
    chan_vese.setExpPhase(new_expPhs)
    new_exp_phase = chan_vese.getExpPhase()
    label_mu.config(text=f"mu = {new_mu}")
    label_nu.config(text=f"nu = {new_nu}")
    label_eta.config(text=f"eta = {new_eta}")
    label_time_step.config(text=f"time_step = {new_time_step}")
    label_epsilon.config(text=f"epsilon = {new_epsilon}")
    label_expPhs.config(text=f"Fases =2^{new_exp_phase}")

def updateParams(event = None):
    setParams(entry_mu.get(),entry_nu.get(),
        entry_eta.get(),
        entry_time_step.get(),
        entry_epsilon.get(),
        entry_expPhs.get()
    )

    updateLambdas()
    
    entry_nu.delete(0, END)
    entry_mu.delete(0, END)
    entry_eta.delete(0, END)
    entry_time_step.delete(0, END)
    entry_epsilon.delete(0, END)
    entry_expPhs.delete(0,END)

def updateReinitialize(event = None):
    restart = entry_reini.get()
    chan_vese.setReinicialize(restart)
    n = chan_vese.getReinicialize()
    entry_reini.delete(0,END)
    if n>0:
        label_reini.config(text=f"Reinicializar a SDF cada {n} iteraciones ")
    else:
        label_reini.config(text=f"Reinicializar a SDF (X) iteraciones")

def updateStop(event = None):
    maxIteraciones = entry_iter.get()
    chan_vese.setMaxIterations(maxIteraciones)
    entry_iter.delete(0, END)
    tol = entry_tol.get()
    chan_vese.setTolerance(tol)
    entry_tol.delete(0, END)
    label_iter.config(text=f"Maximo Iteraciones = {chan_vese.getMaxIterations()}")
    label_tol.config(text=f"Tolerancia = {chan_vese.getTolerance()}")
    chan_vese.setResize(entry_width.get(),entry_height.get())
    resize = chan_vese.getResize()
    label_size.config(text=f"Size(Ancho,Alto)=({resize[0]},{resize[1]})")
    entry_width.delete(0,END)
    entry_height.delete(0,END)
    updateReinitialize()
def selectInitialFunction(nFunction):
    if nFunction ==1:
        bPhi1.set(True)
        bPhi2.set(False)
    else:
        bPhi1.set(False)
        bPhi2.set(True)

    chan_vese.setInitialFunction(nFunction)

def updateExp(event=None):
    updateParams(event)
    updateLambdas(event)

def updateLambdasButton(event=None):
    try:
        for idx, entry in enumerate(frame_lambdas.winfo_children()):
            if isinstance(entry, Entry):
                value = entry.get()
                chan_vese.setLambdas([idx//2, value])
    except ValueError:
        print("Error al intentar actualizar los valores de lambdas")
    updateLambdas()

def updateLambdas(event=None):
    for widget in frame_lambdas.winfo_children():
        widget.destroy()
    try:
        expPhs = chan_vese.getExpPhase()
        num_lambdas = 2 ** expPhs
        lambdas.clear()
        aLambdas = chan_vese.getLambdas()

        if num_lambdas != len(aLambdas):
            chan_vese.setLambdas()
            aLambdas = chan_vese.getLambdas()
        for i in range(num_lambdas):
            label = Label(frame_lambdas, text=f"lambda_{i + 1} = {aLambdas[i]}")
            label.grid(row=i, column=0, padx=10, pady=5)

            entry = Entry(frame_lambdas, width=10)
            entry.delete(0, END)
            entry.bind("<Return>", lambda event, idx=i,e=entry: actualizarLambdaIndividual(idx,e.get()))
            entry.grid(row=i, column=1, padx=10, pady=5)
            lambdas.append(entry)
    except ValueError:
        print("Error: expPhs debe ser un número entero")

def actualizarLambdaIndividual(idx,entry):
    chan_vese.setLambdas([idx,entry])
    updateLambdas()

def call_chan_vese_segmentation():
    chan_vese.chan_vese_segmentation()


root = Tk()
root.title("Chan-Vese Segmentation")
root.config(bg="#F5F5F5")
chan_vese.initializeParams()
icon = Image.open('images/gris_espiral.png')
icon = icon.resize((32, 32))  
root.iconphoto(False, ImageTk.PhotoImage(icon))


# Creación del frame principal
frame_params = Frame(root, bg="#D9D9D9")
frame_params.grid(row=0, column=0,padx=1)

# Frame para mu
frame_container_mu = Frame(frame_params, bg="#D9D9D9")
frame_container_mu.grid(row=0, column=0, sticky="W")
label_mu = Label(frame_container_mu, anchor="w", width=15)
label_mu.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_mu = Entry(frame_container_mu, width=10)
entry_mu.grid(row=0, column=1, pady=5, padx=10, sticky="E")

# Frame para nu
frame_container_nu = Frame(frame_params, bg="#D9D9D9")
frame_container_nu.grid(row=1, column=0, sticky="W")
label_nu = Label(frame_container_nu, anchor="w", width=15)
label_nu.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_nu = Entry(frame_container_nu, width=10)
entry_nu.grid(row=0, column=1, padx=10, pady=5, sticky="E")

# Frame para eta
frame_container_eta = Frame(frame_params, bg="#D9D9D9")
frame_container_eta.grid(row=2, column=0, sticky="W")
label_eta = Label(frame_container_eta,  anchor="w", width=15)
label_eta.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_eta = Entry(frame_container_eta, width=10)
entry_eta.grid(row=0, column=1, padx=10, pady=5, sticky="E")

# Frame para time_step
frame_container_time_step = Frame(frame_params, bg="#D9D9D9")
frame_container_time_step.grid(row=3, column=0, sticky="W")
label_time_step = Label(frame_container_time_step, anchor="w", width=15)
label_time_step.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_time_step = Entry(frame_container_time_step, width=10)
entry_time_step.grid(row=0, column=1, padx=10, pady=5, sticky="E")

# Frame para epsilon
frame_container_epsilon = Frame(frame_params, bg="#D9D9D9")
frame_container_epsilon.grid(row=4, column=0, sticky="W")
label_epsilon = Label(frame_container_epsilon, anchor="w", width=15)
label_epsilon.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_epsilon = Entry(frame_container_epsilon, width=10)
entry_epsilon.grid(row=0, column=1, padx=10, pady=5, sticky="E")

# Frame para Fases
frame_container_expPhs = Frame(frame_params, bg="#D9D9D9")
frame_container_expPhs.grid(row=5, column=0, sticky="W")
label_expPhs = Label(frame_container_expPhs, anchor="w", width=15)
label_expPhs.grid(row=0, column=0, padx=10, pady=5, sticky="W")
entry_expPhs = Entry(frame_container_expPhs, width=10)
entry_expPhs.grid(row=0, column=1, padx=10, pady=5, sticky="E")

# Configuración uniforme de las columnas
frame_params.grid_columnconfigure(0, weight=1)
frame_params.grid_columnconfigure(1, weight=1)


entry_widgets = [entry_mu, entry_nu, entry_eta, entry_time_step, entry_epsilon]

for widget in entry_widgets:
    widget.bind("<Return>", updateParams)
    
entry_expPhs.bind("<Return>", updateExp)


frame_lambdas = Frame(frame_params, bg="#D9D9D9")
frame_lambdas.grid(row=8, column=0)
frame_button_lambda = Frame(frame_params)
frame_button_lambda.grid(row=9,column =0)

button_actualizar = Button(frame_button_lambda, text="Actualizar Lambdas", command=updateLambdasButton, bg ="#dbf8f9")
button_actualizar.grid(row=0, column=0)
lambdas = []
button_set_params = Button(root, text="Actualizar Parámetros del Funcional", command=updateParams, bg="#dbf8f9")
button_set_params.grid(row=1,column=0,pady=5)


# Frame para los parámetros de parada
frame_stop = Frame(root, bg="#D9D9D9")
frame_stop.grid(row=0, column=1, padx=1, sticky="W")

# Label y Entry para Maximo Iteraciones
label_iter = Label(frame_stop, anchor="w", width=20)
label_iter.grid(row=1, column=0, padx=10, pady=5, sticky="W")
entry_iter = Entry(frame_stop, width=10)
entry_iter.grid(row=1, column=1, padx=10, pady=5)

# Label y Entry para Tolerancia
label_tol = Label(frame_stop, anchor="w", width=20)
label_tol.grid(row=2, column=0, padx=10, pady=5, sticky="W")
entry_tol = Entry(frame_stop, width=10)
entry_tol.grid(row=2, column=1, padx=10, pady=5)

# Label para Size (Ancho, Alto)
label_size = Label(frame_stop, anchor="w", width=20)
label_size.grid(row=3, column=0, padx=10, pady=5, sticky="W")

# Frame para los valores de Size (Ancho, Alto)
frame_size = Frame(frame_stop, bg="#D9D9D9")
frame_size.grid(row=3, column=1, sticky="W")
entry_width = Entry(frame_size, width=10)
entry_width.grid(row=0, column=0, pady=5, padx=10, sticky="E")
entry_height = Entry(frame_size, width=10)
entry_height.grid(row=0, column=1, pady=5, padx=10, sticky="E")

#Label iteraciones
label_reini = Label(frame_stop)
label_reini.grid(row=4, column=0, padx=10, pady=5)
entry_reini  = Entry(frame_stop, width=10)
entry_reini.grid(row=4, column=1, padx=10, pady=5)

# Botón para actualizar los parámetros
button_set_stop = Button(root, text="Actualizar Parámetros Secundarios", command=updateStop, bg="#dbf8f9")
button_set_stop.grid(row=1, column=1, pady=5)

# Configuración uniforme de las columnas en el frame de parada
frame_stop.grid_columnconfigure(0, weight=1)
frame_stop.grid_columnconfigure(1, weight=1)
frame_stop.grid_columnconfigure(2, weight=1)  # Asegura que la columna para el tamaño (Ancho, Alto) también esté equilibrada


stop_widgets = [entry_iter,entry_tol,entry_height,entry_width]
for widget in stop_widgets:
    widget.bind("<Return>", updateStop)




bPhi1 = BooleanVar(value = False)
bPhi2 = BooleanVar(value= True)

checkbox_phi1 = Checkbutton(frame_stop, text="Phi 1 (Círculo)", variable=bPhi1, bg="#dbf8f9",
                                 command=lambda :selectInitialFunction(1))
checkbox_phi1.grid(row=0,column=0,padx=5)
checkbox_phi2= Checkbutton(frame_stop, text="Phi 2 (Tablero)", variable=bPhi2, bg="#dbf8f9",
                                 command=lambda :selectInitialFunction(2))
checkbox_phi2.grid(row=0,column=1,padx=5)



checkbox_widgets = [entry_reini]
for widget in checkbox_widgets:
    widget.bind("<Return>", updateReinitialize)


frame_image = Frame(root, bg="#D9D9D9")
frame_image.grid(row=0,column=2,padx=1)
frame_image_title = Label(frame_image, text="Seleccione una imagen", bg="#f0f0f0",fg="black")
frame_image_title.grid(row=0,column=0,pady=2)
labels = []
image_names = list_image_names()

button_segmentation = Button(root, bg="#a6fd9b",
                             command=call_chan_vese_segmentation)
button_segmentation.grid(row=1,column=3,padx=5)

for nName in image_names:
    label = Label(frame_image, text=nName,bg="#D9D9D9",fg="#6c6b6b",cursor="hand2")
    label.grid()
    labels.append(label)
    label.bind("<Button-1>", lambda e, name=nName,lbl=label: on_image_click(name,lbl,labels))
    if nName== "gris_espiral.png": on_image_click("gris_espiral.png",label,labels)

btn_load = Button(root, text="Cargar Imagen", command=load_image_from_PC, bg="#dbf8f9")
btn_load.grid(row=1, column=2)


updateParams()
updateLambdas()
updateStop()
updateReinitialize()
root.pack_propagate(True)
root.mainloop()

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:03:35 2024

convert bin file to mp4; GUI-equipped

@author: Dinghao Luo
"""


#%% imports
import numpy as np  
import sys 
import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from time import time
from datetime import timedelta
import threading  # to seperate the main thread from the processing thread (prevents GUI from freezing)

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
from imaging_utility_functions import gaussian_kernel_unity

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list


#%% global variables
halt_flag = False


#%% functions
def check_GPU(write_func):
    try:
        import cupy as cp 
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
    except ModuleNotFoundError:
        if write_func:
            write_func('cupy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions\n')
        else:
            print('cupy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
        GPU_AVAILABLE = False
    except Exception as e:
        # catch any other unexpected errors and print a general message
        if write_func:
            write_func('An error occurred: {}\n'.format(e))
        else:
            print('An error occurred: {}'.format(e))
        GPU_AVAILABLE = False

    if GPU_AVAILABLE:
        # we are assuming that there is only 1 GPU device
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
        if write_func:
            write_func('GPU-acceleration with {} and cupy\n'.format(str(name)))
        else:
            print('GPU-acceleration with {} and cupy'.format(str(name)))
    else:
        if write_func:
            write_func('GPU-acceleartion unavailable\n')
        else:
            print('GPU-acceleartion unavailable')
        
    return GPU_AVAILABLE


def create_mov(plane_path, out_path, GPU_AVAILABLE, write_func=None, display_func=None, smooth=True, sigma=3, sample_freq=30, first_nsec=300):
    global halt_flag
    
    bin_path = r'{}\data.bin'.format(plane_path)
    ops_path = r'{}\ops.npy'.format(plane_path)
    ops_file = np.load(ops_path, allow_pickle=True).item()
    
    tot_frames = ops_file['nframes']
    shape = tot_frames, ops_file['Ly'], ops_file['Lx']
    
    first_nframes = first_nsec*sample_freq
    if write_func:
        write_func(f'loading first {first_nframes} frames...\n')
    else: 
        print(f'loading first {first_nframes} frames...')
    mov = np.memmap(bin_path, mode='r', dtype='int16', shape=shape)[:first_nframes,:,:]
    if smooth:
        write_func('smoothing through time...\n') if write_func else print('smoothing through time...')
        t0 = time()
        kernel = gaussian_kernel_unity(sigma)
        pad_width = len(kernel) // 2
        if GPU_AVAILABLE:
            import cupy as cp
            kernel = cp.asarray(kernel)
            mov = cp.asarray(mov)
            mov_padded = cp.pad(mov, ((pad_width, pad_width), (0, 0), (0, 0)), mode='reflect')
            mov = cp.apply_along_axis(lambda x: cp.convolve(x, kernel, mode='same'), 
                                      axis=0, 
                                      arr=mov_padded)[pad_width:-pad_width, :, :].get()
        else:
            mov_padded = np.pad(mov, ((pad_width, pad_width), (0, 0), (0, 0)), mode='reflect')
            mov =  np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 
                                       axis=0, 
                                       arr=mov_padded)[pad_width:-pad_width, :, :]
    if write_func:
        write_func('smoothing done ({})\n'.format(str(timedelta(seconds=int(time()-t0)))))
    else:
        print('smoothing done ({})'.format(str(timedelta(seconds=int(time()-t0)))))
    mov_min = mov.min()
    mov_max = mov.max()
    
    # write video to disk
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # will throw warning due to licensing issues, but can ignore
    out = cv2.VideoWriter(out_path, fourcc, 30, (512, 512), isColor=False)
    
    if not out.isOpened():
        write_func('error: could not open video writer\n') if write_func else print('error: could not open video writer')
    else:
        write_func('writing video to disk\n') if write_func else print('writing video to disk')
        for frame in tqdm(range(min(first_nframes, tot_frames)), file=sys.stdout, ncols=100):
            if halt_flag:
                write_func('process halted') if write_func else print('process halted')
                break
                
            frame_data = mov[frame, :, :]
            normalised_frame = ((frame_data - mov_min) / 
                                (mov_max - mov_min) * 255).astype('uint8')  # normalise to all
            frame_bgr = cv2.cvtColor(normalised_frame, cv2.COLOR_GRAY2BGR)  # convert grayscale frame to BGR (3-channel) format
            out.write(frame_bgr)
            
            # convert the frame to an Image format for display
            if display_func:
                pil_image = Image.fromarray(normalised_frame)
                display_func(pil_image)
    
        out.release()
        write_func('writing saved successfully\n') if write_func else print('video saved successfully')


def update_frame_display(pil_image, label, brightness_factor=1.5, contrast_factor=1.5):
    """Update the image in the tkinter Label widget"""
    # enhance the image
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    pil_image_bright = brightness_enhancer.enhance(brightness_factor)
    contrast_enhancer = ImageEnhance.Contrast(pil_image_bright)
    pil_image_contrast = contrast_enhancer.enhance(contrast_factor)
    
    # convert to ImageTk format
    tk_image = ImageTk.PhotoImage(pil_image_contrast)
    
    # update the label with the new image
    label.config(image=tk_image)
    label.image = tk_image  # keep a reference to avoid garbage collection


def run_processing(selected_list, output_text, frame_display_label, out_path_entry):
    global halt_flag
    
    if selected_list == 'HPC-LC-GCaMP':
        paths = rec_list.pathHPCLCGCaMP
    elif selected_list == 'HPC-GRABNE':
        paths = rec_list.pathHPCGRABNE
    else:
        messagebox.showerror('Error', 'not a valid list')
        return
    
    GPU_AVAILABLE = check_GPU(write_func=lambda msg: output_text.insert(tk.END, msg))
    
    for rec_path in paths:
        if halt_flag:
            break
            
        recname = rec_path[-17:]
        
        output_text.insert(tk.END, f'\nprocessing {recname}...\n')
        output_text.yview(tk.END)  # scroll to the latest message
        
        plane_path = r'{}\suite2p\plane0'.format(rec_path)
        
        out_path = r'{}\{}'.format(out_path_entry, recname)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        movie_path = r'{}\movie_ch1.mp4'.format(out_path)
        
        create_mov(plane_path, 
                   movie_path,
                   GPU_AVAILABLE,
                   write_func=lambda msg: output_text.insert(tk.END, msg), 
                   display_func=lambda pil_img: update_frame_display(pil_img, frame_display_label))
        
        output_text.insert(tk.END, f'{recname} done\n')
        output_text.yview(tk.END)
        
    if not halt_flag:
        messagebox.showinfo('Done', 'conversion complete')
    else:
        messagebox.showinfo('Halted', 'conversion process halted')


def browse_out_path(out_path_entry):
    """Open a directory dialog and update the out_path entry"""
    out_path = filedialog.askdirectory(title='Select Output Directory')
    if out_path:  # if a directory is selected
        out_path_entry.delete(0, tk.END)  # clear the current entry
        out_path_entry.insert(0, out_path)  # insert the selected directory


def halt_processing():
    global halt_flag
    halt_flag = True


def start_processing_thread(selected_list, output_text, frame_display_label, out_path_entry):
    # run the processing function in a separate thread to avoid freezing the GUI
    processing_thread = threading.Thread(target=run_processing, args=(selected_list, output_text, frame_display_label, out_path_entry))
    processing_thread.start()


def create_gui():
    # create main window
    window = tk.Tk()
    window.title('Recording Conversion')
    
    # configure grid columns
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=0)
    
    # provide a dropdown menu to select the recording list
    label = ttk.Label(window, text='Select Recording List:')
    label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    list_options = ['HPC-LC-GCaMP', 'HPC-GRABNE']
    list_selector = ttk.Combobox(window, values=list_options)
    list_selector.grid(row=0, column=1, padx=10, pady=10, sticky='w')
    
    # terminal
    output_text = tk.Text(window, height=10, width=50, wrap=tk.WORD)
    output_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    output_text.insert(tk.END, '')
    
    # create label widget for displaying frames
    frame_display_label = ttk.Label(window)
    frame_display_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    # create entry for the output path
    out_path_label = ttk.Label(window, text='Select Out Path:')
    out_path_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')
    
    out_path_entry = ttk.Entry(window)
    out_path_entry.grid(row=4, column=0, padx=10, pady=10, sticky='ew')  # Make it expand horizontally
    
    # Configure row 4 to expand horizontally as well
    window.grid_rowconfigure(4, weight=1)  # Allow the row to expand
    
    # create a button to open directory picker
    browse_button = ttk.Button(window, text='Browse', command=lambda: browse_out_path(out_path_entry))
    browse_button.grid(row=4, column=1, padx=10, pady=10, sticky='w')
    
    # button frame to contain the 2 buttons
    button_frame = ttk.Frame(window)
    button_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky='w')  # Place frame in grid
    
    # run button to start processing
    run_button = ttk.Button(button_frame, text='Run', command=lambda: start_processing_thread(list_selector.get(), output_text, frame_display_label, out_path_entry))
    run_button.grid(row=0, column=0, padx=10, pady=10)  # Place Run button in the frame
    
    # halt button to stop processing
    halt_button = ttk.Button(button_frame, text='Halt', command=halt_processing)
    halt_button.grid(row=0, column=1, pady=10)  # Place Halt button next to Run button
    
    window.mainloop()
    

#%% Run the GUI
if __name__ == '__main__':
    create_gui()
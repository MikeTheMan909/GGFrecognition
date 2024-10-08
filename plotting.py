import numpy as np
import matplotlib.pyplot as plt

def plott_shapes(data_arrays, labels, colors):
    num_datasets = len(data_arrays)
    plt.figure(figsize=(10, 6))
    for j in range(num_datasets):
        num_images = len(data_arrays[j])  # Get the length of the current dataset's color channel
        photo_numbers = np.arange(1, num_images + 1)  # Create x-axis values corresponding to the number of images
        
        plt.plot(photo_numbers, data_arrays[j], marker='o', color=colors[j], label=labels[j])
    plt.xlabel("Photo number")
    plt.ylabel("Shapes")
    plt.title(f"Shapes across Photos")
    plt.legend()
    plt.show()


def plotting(data_arrays, labels, colors):    
    num_datasets = len(data_arrays)  # Number of datasets
    
    # Titles for each plot (Red, Green, Blue, Hue, Saturation, Brightness)
    titles = ['Red Value', 'Green Value', 'Blue Value', 'Hue Value', 'Saturation Value', 'Brightness Value']
    
    # Loop through each color channel (red, green, blue, hue, saturation, value)
    for i in range(6):
        plt.figure(figsize=(10, 6))
        
        # Plot each dataset's corresponding color channel
        for j in range(num_datasets):
            num_images = len(data_arrays[j][i])  # Get the length of the current dataset's color channel
            photo_numbers = np.arange(1, num_images + 1)  # Create x-axis values corresponding to the number of images
            
            plt.plot(photo_numbers, data_arrays[j][i], marker='o', color=colors[j], label=labels[j])
        
        plt.xlabel("Photo number")
        plt.ylabel(titles[i])
        plt.title(f"{titles[i]} across Photos")
        plt.legend()
        plt.show()


import os

def rename_pngs(directory):
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter out only the .png files
    png_files = [f for f in files if f.lower().endswith('.png')]
    # = [f for f in files if f.lower().endswith('.txt')]
    
    # Sort the list to maintain an order
    png_files.sort()
    #txt_files.sort()

    # Rename each file to a sequential number
    for index, filename in enumerate(png_files, start=1):
        # Get the full path of the current file
        old_path = os.path.join(directory, filename)
        
        # Define the new filename
        new_filename = f"{index}.png"
        new_path = os.path.join(directory, new_filename)
        
        counter = index
        while os.path.exists(new_path):
            counter += 1
            new_filename = f"{counter}.png"
            new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

def rename_txt(directory):
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter out only the .png files
    txt_files = [f for f in files if f.lower().endswith('.txt')]
    # = [f for f in files if f.lower().endswith('.txt')]
    
    # Sort the list to maintain an order
    txt_files.sort()
    #txt_files.sort()

    # Rename each file to a sequential number
    for index, filename in enumerate(txt_files, start=1):
        # Get the full path of the current file
        old_path = os.path.join(directory, filename)
        
        # Define the new filename
        new_filename = f"{index}.txt"
        new_path = os.path.join(directory, new_filename)
        
        counter = index
        while os.path.exists(new_path):
            counter += 1
            new_filename = f"{counter}.png"
            new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")
# Example usage
rename_pngs('C:\\Users\\Jackw\\Documents\\GitHub\\openGL_floorplan\\pascal data')
rename_txt('C:\\Users\\Jackw\\Documents\\GitHub\\openGL_floorplan\\pascal data')
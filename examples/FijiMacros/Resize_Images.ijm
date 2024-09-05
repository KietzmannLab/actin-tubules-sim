// Set the directory containing the images to resize
inputDir = getDirectory("Choose Directory with 128x128 images");
list = getFileList(inputDir);

// Set the directory where resized images will be saved
outputDir = getDirectory("Choose Directory to Save Resized Images");

// Loop through all images in the directory
for (i = 0; i < list.length; i++) {
    // Open the current image
    open(inputDir + list[i]);
    
    // Check if the image is 128x128, otherwise skip resizing
    width = getWidth();
    height = getHeight();
    
  
        // Resize the image to 256x256
        run("Size...", "width=256 height=256 depth=1 constrain average interpolation=Bilinear");
        
        // Save the resized image in the output directory with "_resized" suffix
        saveAs("Tiff", outputDir + list[i] + "_resized.tif");
    

    // Close the image to save memory
    close();
}

// End of macro

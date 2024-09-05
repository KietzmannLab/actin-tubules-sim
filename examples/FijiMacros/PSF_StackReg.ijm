// Set the directory containing PSF files
dir = getDirectory("Choose Directory with PSF files");
list = getFileList(dir);

// Open the first image as the reference for registration
open(dir + list[0]);
imageSum = getImageID(); // Save reference to the sum image

// Loop through all PSF files in the directory starting from the second image
for (i = 1; i < list.length; i++) {
    // Open the current PSF file
    open(dir + list[i]);
    currentImage = getImageID();
    
    // Register the current image to the first image (using translation only)
    run("StackReg", "transformation=Translation");

    // Add the registered image to the sum
    imageCalculator("Add stack", "ID="+imageSum, "ID="+currentImage);
    
    // Close the current image to save memory
    close();
}

// Calculate the average PSF by dividing by the number of images
selectImage(imageSum);
run("Multiply...", "value=" + 1/list.length);

// Save the averaged PSF if desired
saveAs("Tiff", dir + "averaged_PSF_registered.tif");

// Display the averaged PSF
run("Enhance Contrast", "saturated=0.35");

// End of macro
#include <tiffio.h>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2) {
        cerr << "Syntax: app filepath" << endl;
        return EXIT_FAILURE;
    }

    const string filePath(argv[1]);

    // Open the TIFF file using libtiff
    TIFF* tif = TIFFOpen(filePath.c_str(), "r");
    if (!tif) {
        cerr << "Error opening TIFF" << endl;
        return EXIT_FAILURE;
    }

    do {
        unsigned int width, height;
        uint32* raster;

        // get the size of the tiff
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        auto scanlength = TIFFScanlineSize(tif);

        std::uint32_t npixels = width * height;  // get the total number of pixels

        cout << "Width: " << width << endl;
        cout << "Height: " << height << endl;
        cout << "scanlength: " << scanlength << endl;

        raster = (uint32*)_TIFFmalloc(npixels * sizeof(uint32));
        // allocate temp memory (must use the tiff library malloc)
        if (raster == NULL)
        {
            TIFFClose(tif);
            cerr << "Could not allocate memory for raster of TIFF image" << endl;
            return EXIT_FAILURE;
        }

        // Check the tif read to the raster correctly
        if (!TIFFReadRGBAImage(tif, width, height, raster, 0)) {
            TIFFClose(tif);
            cerr << "Could not read raster of TIFF image" << endl;
            return EXIT_FAILURE;
        }

        // itterate through all the pixels of the tif
        for (std::uint32_t x = 0; x < width; x++)
            for (std::uint32_t y = 0; y < height; y++) {
                uint32& TiffPixel = raster[y * width + x];  // read the current pixel of the TIF
            }

        _TIFFfree(raster);             // release temp memory
    } while (TIFFReadDirectory(tif));  // get the next tif
    TIFFClose(tif);                    // close the tif file
    return 0;
}

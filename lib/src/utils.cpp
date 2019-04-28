#include <lib/utils.h>

#include <jpeglib.h>
#include <setjmp.h>

#include <stdexcept>

struct my_error_mgr
{
    struct jpeg_error_mgr pub; /* "public" fields */

    jmp_buf setjmp_buffer; /* for return to caller */
};

typedef struct my_error_mgr* my_error_ptr;

/*
 * Here's the routine that will replace the standard error_exit method:
 */
METHODDEF(void) my_error_exit(j_common_ptr cinfo)
{
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    my_error_ptr myerr = (my_error_ptr)cinfo->err;

    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message)(cinfo);

    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
}

std::vector<unsigned char> read_JPEG_file(const std::string& path, std::uint32_t& width, std::uint32_t& height)
{
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    FILE* infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(path.c_str(), "rb")) == NULL) {
        throw std::runtime_error("Coult not open " + path);
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        throw std::runtime_error("Error decompressing");
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    std::vector<unsigned char> image(row_stride * cinfo.output_height);
    int row = 0;

    while (cinfo.output_scanline < cinfo.output_height) {
        /* jpeg_read_scanlines expects an array of pointers to scanlines.
         * Here the array is only one element long, but you could ask for
         * more than one scanline at a time if that's more convenient.
         */
        jpeg_read_scanlines(&cinfo, buffer, 1);
        std::copy(buffer[0], buffer[0] + row_stride, &image.data()[row * row_stride]);
        row++;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    width = cinfo.output_width;
    height = cinfo.output_height;
    return image;
}

/// \todo adapt this method to handle greyscale or color images
void write_JPEG_file(const std::vector<unsigned char>& image, const std::string& path, std::uint32_t width,
                     std::uint32_t height, std::uint8_t quality)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile;           /* target file */
    JSAMPROW row_pointer[1]; /* pointer to JSAMPLE row[s] */
    int row_stride;          /* physical row width in image buffer */

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(path.c_str(), "wb")) == NULL) {
        std::runtime_error("can't open " + path);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width; /* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = 1;     /* # of color components per pixel */
//    cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
    cinfo.in_color_space = JCS_GRAYSCALE; /* colorspace of input image */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 1; /* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        /* jpeg_write_scanlines expects an array of pointers to scanlines.
         * Here the array is only one element long, but you could pass
         * more than one scanline at a time if that's more convenient.
         */
        row_pointer[0] = const_cast<JSAMPLE *>(&image.data()[cinfo.next_scanline * row_stride]);
        (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

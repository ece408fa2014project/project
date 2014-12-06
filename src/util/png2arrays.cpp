#include "png2arrays.h"

png2arrays::~png2arrays() {
    delete[] this->r;
    delete[] this->g;
    delete[] this->b;
}

void png2arrays::parse_png(const PNG * image) {
    int size = image->width() * image->height();
    this->x_dim = image->width();
    this->y_dim = image->height();
    this->r = new float[size];
    this->g = new float[size];
    this->b = new float[size];
    for(int i = 0; i < image->height(); i++)
        for(int j = 0; j < image->width(); j++)
        {
            const RGBAPixel * pix = (*image)(j, i);
            r[i * image->width() + j] = (float) pix->red;
            g[i * image->width() + j] = (float) pix->green;
            b[i * image->width() + j] = (float) pix->blue;
        }
}

PNG * png2arrays::from_arrays() {
    PNG * ret = new PNG(x_dim, y_dim);

    for(int i = 0; i < this->y_dim; i++)
    {
        for(int j = 0; j < this->x_dim; j++)
        {
            (*ret)(j, i)->red = (unsigned char) r[i * x_dim + j];
            (*ret)(j, i)->green = (unsigned char) g[i * x_dim + j];
            (*ret)(j, i)->blue = (unsigned char) b[i * x_dim + j];
        }
    }

    return ret;
}

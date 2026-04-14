#include <iostream>
#include "Image2D.h"
#include "Sampling.h"
#include "BitStream.h"
#include "RLE.h"
#include "DCT.h"
#include "Segments.h"
#include "Quantization.h"
#include "utils.h"

int main()
{
    Image2D<uint8_t> img = loadPPM("../Samples/TestPPM.ppm");
    img.setSteps(2, 2);
    Image2D<float> yCbCr = RGBtoYCbCr<float>(img, false);
    yCbCr.setSteps(8, 8);
    yCbCr.setSteps(2, 2); // 4:2:0 sampling method
    local_avg_parallel(yCbCr);


    auto& Y  = yCbCr.getChannel1();
    dctDirectImp(Y,  Y.rows(),  Y.cols());
    quantize_channel(Y, Y.rows(), Y.cols(), jpeg_luma_quantization_table);

    auto& Cb = yCbCr.getChannel2();
    dctDirectImp(Cb, Cb.rows(), Cb.cols());
    quantize_channel(Cb, Cb.rows(), Cb.cols(), jpeg_chroma_quantization_table);

    auto& Cr = yCbCr.getChannel3();
    dctDirectImp(Cr, Cr.rows(), Cr.cols());
    quantize_channel(Cr, Cr.rows(), Cr.cols(), jpeg_chroma_quantization_table);

    auto& Y1  = yCbCr.getChannel1();
    auto& Cb1 = yCbCr.getChannel2();
    auto& Cr1 = yCbCr.getChannel3();

    std::cout << "sx=" << yCbCr.sx << " sy=" << yCbCr.sy << "\n";
    std::cout << "Y : "  << Y1.rows()  << "x" << Y1.cols()
              << " blocks=" << (Y1.rows()/8)*(Y1.cols()/8) << "\n";
    std::cout << "Cb: "  << Cb1.rows() << "x" << Cb1.cols()
              << " blocks=" << (Cb1.rows()/8)*(Cb1.cols()/8) << "\n";
    std::cout << "Cr: "  << Cr1.rows() << "x" << Cr1.cols()
              << " blocks=" << (Cr1.rows()/8)*(Cr1.cols()/8) << "\n";


    auto bitStreamData = generate_bit_stream_data_for_image(yCbCr);


    BitStream bitstream;
    Segments segments (yCbCr, bitstream);
    segments.writeStartEnd(true);
    segments.writeAPP0();
    segments.writeDQT();
    segments.writeSOF0();

    segments.writeDHT(y_dc_tree, false, 0);
    segments.writeDHT(y_ac_tree, true,  0);
    segments.writeDHT(c_dc_tree, false, 1);
    segments.writeDHT(c_ac_tree, true,  1);

    segments.writeSOS();
    bitstream.byteStuffingEnabled = true;
    for (size_t i = 0; i < bitStreamData.size(); ++i) {
        const auto& bsd = bitStreamData[i];

        if (bsd.huffman_code_length < 0 || bsd.huffman_code_length > 64 ||
            bsd.code_length < 0 || bsd.code_length > 64) {
            std::cerr << "BAD length at i=" << i
                      << " hlen=" << bsd.huffman_code_length
                      << " clen=" << bsd.code_length
                      << " huff=" << bsd.huffman_code
                      << " code=" << bsd.code << "\n";
            break;
            }
        bitstream.writeBitsBackFast(bsd.huffman_code, (uint32_t)bsd.huffman_code_length);
        if (bsd.code_length > 0) {
            bitstream.writeBitsBackFast(bsd.code, (uint32_t)bsd.code_length);
        }
    }
    bitstream.flushBack();

    bitstream.byteStuffingEnabled = false;
    segments.writeStartEnd(false);
    bitstream.writeToFile("result.jpg");

    return 0;
}

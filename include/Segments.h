#ifndef JPEG_PROJEKT_SEGMENTS_H
#define JPEG_PROJEKT_SEGMENTS_H
#include "BitStream.h"
#include "Image2D.h"
#include "Quantization.h"

template <typename T>
class Segments
{
public:
    Image2D<T>& img;
    BitStream& bitstream;

    Segments(Image2D<T>& image, BitStream& bs)
        : img(image), bitstream(bs) {}


    void writeAPP0()
    {
        std::vector<uint8_t> buf;

        // Marker
        buf.push_back(0xFF);
        buf.push_back(0xE0);

        // Length placeholder
        buf.push_back(0x00);
        buf.push_back(0x00);

        // JFIF
        buf.push_back(0x4A);
        buf.push_back(0x46);
        buf.push_back(0x49);
        buf.push_back(0x46);
        buf.push_back(0x00);

        // Version
        buf.push_back(0x01);
        buf.push_back(0x01);

        // Density
        buf.push_back(0x00);
        buf.push_back(0x00);
        buf.push_back(0x01);
        buf.push_back(0x00);
        buf.push_back(0x01);

        // No Preview
        buf.push_back(0x00);
        buf.push_back(0x00);

        // Length
        uint16_t length = static_cast<uint16_t>(buf.size() - 2);

        buf[2] = (length >> 8) & 0xFF;
        buf[3] = length & 0xFF;

        for (uint8_t b : buf)
            bitstream.writeBitsBackFast(b, 8);
    }

    void writeDQT()
    {
        std::vector<uint8_t> buf;

        // Marker
        buf.push_back(0xFF);
        buf.push_back(0xDB);

        // Placeholder length
        buf.push_back(0x00);
        buf.push_back(0x00);

        // Quantization table
        auto appendQT8 = [&](uint8_t tableId, const auto& qt)
        {
            uint8_t info = (0x00) | (tableId & 0x0F);
            buf.push_back(info);

            std::vector<int> zz = zigzag_order_block(qt);
            for (int v : zz) {
                int clamped = std::clamp(v, 1, 255);
                buf.push_back(static_cast<uint8_t>(clamped));
            }
        };

        appendQT8(0, jpeg_luma_quantization_table);
        appendQT8(1, jpeg_chroma_quantization_table);

        // Length
        uint16_t length = static_cast<uint16_t>(buf.size() - 2);
        buf[2] = (length >> 8) & 0xFF;
        buf[3] = length & 0xFF;

        for (uint8_t b : buf)
            bitstream.writeBitsBackFast(b, 8);
    }

    void writeSOF0()
    {
        std::vector<uint8_t> buf;

        // Marker
        buf.push_back(0xFF);
        buf.push_back(0xC0);

        // Placeholder length
        buf.push_back(0x00);
        buf.push_back(0x00);

        // Precision of data
        buf.push_back(0x08);

        // Resolution
        uint16_t height = static_cast<uint16_t>(img.height);
        uint16_t width = static_cast<uint16_t>(img.width);
        buf.push_back((height >> 8) & 0xFF);
        buf.push_back(height & 0xFF);
        buf.push_back((width >> 8) & 0xFF);
        buf.push_back(width & 0xFF);

        // Number of components
        buf.push_back(0x03);

        // Calculation of subsampling
        uint8_t H = std::clamp(img.sx, 1, 4);
        uint8_t V = std::clamp(img.sy, 1, 4);
        uint8_t sampling = (H << 4) | V;

        // Y
        buf.push_back(0x01);
        buf.push_back(sampling);
        buf.push_back(0x00);

        // Cb
        buf.push_back(0x02);
        buf.push_back(0x11);
        buf.push_back(0x01);

        // Cr
        buf.push_back(0x03);
        buf.push_back(0x11);
        buf.push_back(0x01);

        // Length
        uint16_t length = static_cast<uint16_t>(buf.size() - 2); // ab Längefeld bis Ende, Marker nicht Teil

        buf[2] = (length >> 8) & 0xFF;
        buf[3] = length & 0xFF;

        for (uint8_t b : buf)
            bitstream.writeBitsBackFast(b, 8);
    }

    template <typename SymT>
    void writeDHT(Builder<SymT>& b, bool isAC = false, uint8_t tableID = 0)
    {
        std::vector<uint8_t> buf;
        buf.push_back(0xFF);
        buf.push_back(0xC4);

        buf.push_back(0x00); // length hi (placeholder)
        buf.push_back(0x00); // length lo

        uint8_t tableInfo = (isAC ? 0x10 : 0x00) | (tableID & 0x0F);
        buf.push_back(tableInfo);

        uint8_t Li[16] = {0};

        struct Entry {
            uint8_t  sym;
            uint32_t len;
            uint32_t code;
        };
        std::vector<Entry> entries;
        entries.reserve(b.codeMap.size());

        auto to_u8 = [](const SymT& s) -> uint8_t {
            if constexpr (std::is_same_v<SymT, std::byte>)
                return static_cast<uint8_t>(std::to_integer<unsigned char>(s));
            else
                return static_cast<uint8_t>(s);
        };

        for (auto& [symKey, encLen] : b.codeMap)
        {
            uint32_t code = encLen.first;
            uint32_t len  = encLen.second;

            // Baseline: 1..16
            if (len == 0 || len > 16) {
                std::cerr << "Invalid Huffman code length: " << len << std::endl;
                std::exit(1);
            }

            Li[len - 1]++;

            entries.push_back(Entry{
                to_u8(symKey),
                len,
                code
            });
        }

        for (int i = 0; i < 16; i++)
            buf.push_back(Li[i]);

        // IMPORTANT: Ordering (len, code)
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.len != b.len) return a.len < b.len;
            return a.code < b.code;
        });

        for (auto& e : entries)
            buf.push_back(e.sym);

        uint16_t length = static_cast<uint16_t>(buf.size() - 2);
        buf[2] = (length >> 8) & 0xFF;
        buf[3] = length & 0xFF;

        for (uint8_t v : buf)
            bitstream.writeBitsBackFast(v, 8);
    }


    void writeSOS()
    {
        std::vector<uint8_t> buf;

        // Marker
        buf.push_back(0xFF);
        buf.push_back(0xDA);

        // Length: 6 + 2 * 3 = 12 bytes (for 3 components).
        buf.push_back(0x00);
        buf.push_back(0x0C);

        // Number of components
        buf.push_back(0x03);

        // Y: ID=1, DC=0, AC=0
        buf.push_back(0x01);
        buf.push_back(0x00);

        // Cb: ID=2, DC=1, AC=1
        buf.push_back(0x02);
        buf.push_back(0x11);

        // Cr: ID=3, DC=1, AC=1
        buf.push_back(0x03);
        buf.push_back(0x11);

        buf.push_back(0x00);
        buf.push_back(0x3F);
        buf.push_back(0x00);

        for (uint8_t b : buf)
            bitstream.writeBitsBackFast(b, 8);
    }



    void writeStartEnd(bool start) {
        if (start)
            bitstream.writeBitsBackFast(0xffd8, 16);
        else
            bitstream.writeBitsBackFast(0xffd9, 16);
    }
};

#endif //JPEG_PROJEKT_SEGMENTS_H
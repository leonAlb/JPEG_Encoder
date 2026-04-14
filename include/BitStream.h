#ifndef JPEG_PROJEKT_BITSTREAM_H
#define JPEG_PROJEKT_BITSTREAM_H
#include <vector>
#include <fstream>
#include <iostream>
#include "Eigen/Core"

class BitStream {
private:
    uint8_t bitPositionBack = 0;
    std::byte currentByteBack{0};

public:
    std::vector<std::byte> buffer;
    uint32_t readIndex = 0;
    uint8_t readBitPos = 0;
    bool byteStuffingEnabled = false;

    void pushByte(std::byte b) {
        buffer.push_back(b);
        if (byteStuffingEnabled && b == std::byte{0xFF}) {
            buffer.push_back(std::byte{0x00});
        }
    }


    bool hasBits() const {
        size_t totalBits = buffer.size() * 8;
        size_t currentBit = readIndex * 8 + readBitPos;
        return currentBit < totalBits;
    }


    void writeBit(bool bit) {
        currentByteBack |= static_cast<std::byte>(bit) << (7 - bitPositionBack);
        if (++bitPositionBack == 8) {
            pushByte(currentByteBack);
            currentByteBack = std::byte{0};
            bitPositionBack = 0;
        }
    }


    void writeBitsBackFast(uint64_t bits, uint32_t count) {
        while (count > 0) {
            const uint8_t remaining = static_cast<uint8_t>(8 - bitPositionBack);
            const uint8_t writeNow  = std::min<uint8_t>(remaining, static_cast<uint8_t>(count));
            const uint8_t shift     = static_cast<uint8_t>(remaining - writeNow);

            const uint64_t mask  = (writeNow == 64) ? ~0ULL : ((1ULL << writeNow) - 1ULL);
            const uint64_t chunk = (bits >> (count - writeNow)) & mask;

            currentByteBack |= static_cast<std::byte>(static_cast<uint8_t>(chunk << shift));

            bitPositionBack = static_cast<uint8_t>(bitPositionBack + writeNow);
            count -= writeNow;

            if (bitPositionBack == 8) {
                pushByte(currentByteBack);
                currentByteBack = std::byte{0};
                bitPositionBack = 0;
            }
        }
    }


    bool readBit() {
        if (readIndex >= buffer.size())
            throw std::out_of_range("End of bitstream");

        bool bit = (std::to_integer<uint8_t>(buffer[readIndex]) >> (7 - readBitPos)) & 1u;

        if (++readBitPos == 8) {
            readBitPos = 0;
            readIndex++;
        }
        return bit;
    }

    //max 64 bits lesbar
    uint64_t readBitsFast(uint32_t count)
    {
        uint64_t bits = 0;
        while (count > 0) {
            if (readIndex >= buffer.size()) {
                throw std::out_of_range("End of bitstream");
            }
            uint8_t bitInByte = 8 - readBitPos;
            uint8_t take = std::min<uint32_t>(bitInByte, count);
            uint8_t curByte = std::to_integer<uint8_t>(buffer[readIndex]);
            uint8_t extracted = (curByte >> (bitInByte - take)) & ((1u << take) - 1);
            bits = (bits << take) | extracted;

            readBitPos += take;
            if (readBitPos == 8) {
                readBitPos = 0;
                readIndex++;
            }
            count -= take;
        }
        return bits;
    }

    void flushBack() {
        if (bitPositionBack > 0) {
            uint8_t mask = (0xFF >> bitPositionBack);
            currentByteBack |= static_cast<std::byte>(mask);
            pushByte(currentByteBack);
            currentByteBack = std::byte{0};
            bitPositionBack = 0;
        }
    }


    void writeToFile(const std::string &filename) {
        flushBack();
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Fehler beim Öffnen (write)");
        out.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    }

    void writeBitsAsText(const std::string &filename) {
        flushBack();
        std::ofstream out(filename);
        for (auto b : buffer) {
            uint8_t byte = static_cast<uint8_t>(b);
            for (int i = 7; i >= 0; i--) {
                out << ((byte >> i) & 1 ? '1' : '0');
            }
        }
    }

    void readFromFile(const std::string &filename) {
        std::ifstream in(filename, std::ios::binary | std::ios::ate);
        if (!in)
            throw std::runtime_error("Fehler beim Öffnen (read)");

        auto fileSize = in.tellg();
        in.seekg(0, std::ios::beg);

        buffer.resize(fileSize);
        in.read(reinterpret_cast<char *>(buffer.data()), fileSize);

        readIndex = 0;
        readBitPos = 0;
    }

    void testBSOV() {
        int N = 10000000;

        // Write
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i)
            writeBit(i & 1);
        flushBack();
        writeToFile("bitstream.bin");
        auto mid = std::chrono::high_resolution_clock::now();

        // Read
        readFromFile("bitstream.bin");
        size_t count = 0;
        while (true) {
            try {
                readBit();
                ++count;
            } catch (...) {
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Write time: "
                << std::chrono::duration<double, std::milli>(mid - start).count() << " ms\n";
        std::cout << "Read time: "
                << std::chrono::duration<double, std::milli>(end - mid).count() << " ms\n";
        std::cout << "Bits read: " << count << "\n";
    }
};

#endif //JPEG_PROJEKT_BITSTREAM_H

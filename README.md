# jpeg_projekt

C++20 project that implements a JPEG-style encoding pipeline.
The current demo program reads a text-based PPM image (P3) and writes a baseline JPEG file.

## Highlights

- RGB → YCbCr conversion
- Chroma subsampling (configured in `main.cpp`)
- 8×8 block DCT + quantization (standard JPEG tables)
- Zig-zag scan + run-length encoding (RLE)
- Huffman coding + JPEG marker/segment writer (APP0, DQT, SOF0, DHT, SOS)

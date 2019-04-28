#pragma once

#include <vector>
#include <string>

std::vector<unsigned char> read_JPEG_file(const std::string &path, std::uint32_t& width, std::uint32_t& height);

void write_JPEG_file(const std::vector<unsigned char>& image, const std::string& path, std::uint32_t width,
                     std::uint32_t height, std::uint8_t quality);

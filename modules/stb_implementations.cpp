#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Mip chain generation for the KTX converter (modules/texture_converter.cpp).
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

// Font atlas baking for the custom GUI (modules/gui_font.hpp). rect_pack comes first because
// stb_truetype's pack API picks up the real packer when STB_RECT_PACK_VERSION is already defined,
// and falls back to a much worse internal one when it isn't.
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

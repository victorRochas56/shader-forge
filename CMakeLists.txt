cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Auto-detect vcpkg toolchain
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Try environment variable first
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    # Try relative path (if vcpkg is in project)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    endif()
endif()

# Auto-detect triplet
if(NOT DEFINED VCPKG_TARGET_TRIPLET AND NOT DEFINED ENV{VCPKG_TARGET_TRIPLET})
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows")
    elseif(APPLE)
        set(VCPKG_TARGET_TRIPLET "x64-osx")
    else()
        set(VCPKG_TARGET_TRIPLET "x64-linux")
    endif()
endif()

if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "SLANGC_EXECUTABLE not specified. Please set it with -DSLANGC_EXECUTABLE=<path>")
endif()

#SHADERS
set(SHADER_SLANG_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang"
)

function(add_slang_shader_target TARGET EXECUTABLE_TARGET)
    cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(TEXTURES_DIR ${CMAKE_CURRENT_LIST_DIR}/textures)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)
    
    # Create build-time shaders directory
    add_custom_command(
        OUTPUT ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
    
    # Get all .slang files in the directory
    file(GLOB SLANG_FILES "${SHADERS_DIR}/*.slang")

    # Generate output filenames 
    set(SLANG_SPV_FILES)
    foreach(SLANG_FILE ${SLANG_FILES})
        get_filename_component(FILENAME ${SLANG_FILE} NAME_WE)
        list(APPEND SLANG_SPV_FILES "${SHADERS_DIR}/${FILENAME}.spv")
    endforeach()

    #compile each .slang to corresponding .spv
    foreach(SLANG_FILE ${SLANG_FILES})
        get_filename_component(FILENAME ${SLANG_FILE} NAME_WE)
        set(OUTPUT_FILE "${SHADERS_DIR}/${FILENAME}.spv")
        
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${SLANGC_EXECUTABLE} ${SLANG_FILE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -o ${FILENAME}.spv
            WORKING_DIRECTORY ${SHADERS_DIR}
            DEPENDS ${SLANG_FILE}
            COMMENT "Compiling ${FILENAME}.slang"
            VERBATIM
        )
    endforeach()

    # Get all .slang files and generate corresponding .spv filenames
    file(GLOB SLANG_FILES "${SHADERS_DIR}/*.slang")
    set(SLANG_SPV_FILES)
    foreach(SLANG_FILE ${SLANG_FILES})
        get_filename_component(FILENAME ${SLANG_FILE} NAME_WE)
        list(APPEND SLANG_SPV_FILES "${SHADERS_DIR}/${FILENAME}.spv")
    endforeach()

    # post-build step to copy all compiled shaders and textures to executable directory
    add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/shaders
        COMMENT "Copying compiled shaders to executable directory"
        VERBATIM
    )

    # Copy compiled shaders
    foreach(SPV_FILE ${SLANG_SPV_FILES})
        get_filename_component(FILENAME ${SPV_FILE} NAME)
        add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV_FILE} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/shaders/${FILENAME}
            VERBATIM
        )
    endforeach()

    # Copy textures
    add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/textures
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEXTURES_DIR} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/textures
        COMMENT "Copying textures to executable directory"
        VERBATIM
    )
    
    add_custom_target(${TARGET} DEPENDS ${SLANG_SPV_FILES})
endfunction()

project(shader-forge)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VULKAN_INCLUDE_DIR}
)

# Find packages
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Stb REQUIRED) #no config
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Ktx CONFIG REQUIRED) # !upper case K

add_executable(shader-forge
    main.cpp
)

if(MSVC)
    target_compile_options(shader-forge PRIVATE /std:c++20 /Zc:__cplusplus)
endif()

# Link libraries
target_link_libraries(shader-forge PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    nlohmann_json::nlohmann_json
    KTX::ktx
)

target_include_directories(shader-forge PRIVATE ${Stb_INCLUDE_DIR})

add_slang_shader_target(slang-c shader-forge SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(shader-forge slang-c)

get_target_property(COMPILE_FLAGS shader-forge COMPILE_OPTIONS)
message(STATUS "Target compile options: ${COMPILE_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
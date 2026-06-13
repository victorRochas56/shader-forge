cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

#builds with MSCV and uses VCPCKG to fetch some dependencies
#also copies all resources (textures, models ... etc.) to the build directory

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
    set(MODELS_DIR ${CMAKE_CURRENT_LIST_DIR}/models)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)
   
    # Create build-time shaders directory
    add_custom_command(
        OUTPUT ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
   
    # Get compilable .slang files (top-level only, modules/ is excluded by the glob pattern)
    file(GLOB SLANG_FILES "${SHADERS_DIR}/*.slang")
    # Get all module files as dependencies (any .slang in modules/ subfolder)
    file(GLOB SLANG_MODULES "${SHADERS_DIR}/modules/*.slang")
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
            COMMAND ${SLANGC_EXECUTABLE} ${SLANG_FILE} -target spirv -profile spirv_1_4 -fvk-use-entrypoint-name -g -o ${FILENAME}.spv
            WORKING_DIRECTORY ${SHADERS_DIR}
            DEPENDS ${SLANG_FILE} ${SLANG_MODULES}
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
    
    # Copy compiled shaders (only if different)
    foreach(SPV_FILE ${SLANG_SPV_FILES})
        get_filename_component(FILENAME ${SPV_FILE} NAME)
        add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_FILE} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/shaders/${FILENAME}
            VERBATIM
        )
    endforeach()
    
    # Copy textures (only if different)
    file(GLOB TEXTURE_FILES "${TEXTURES_DIR}/*")
    add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/textures
        COMMENT "Copying textures to executable directory"
        VERBATIM
    )
    foreach(TEXTURE_FILE ${TEXTURE_FILES})
        get_filename_component(FILENAME ${TEXTURE_FILE} NAME)
        if(IS_DIRECTORY ${TEXTURE_FILE})
            add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${TEXTURE_FILE} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/textures/${FILENAME}
                VERBATIM
            )
        else()
            add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEXTURE_FILE} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/textures/${FILENAME}
                VERBATIM
            )
        endif()
    endforeach()
    
    # Copy models (only if different)
    file(GLOB MODEL_FILES "${MODELS_DIR}/*")
    add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/models
        COMMENT "Copying models to executable directory"
        VERBATIM
    )
    foreach(MODEL_FILE ${MODEL_FILES})
        get_filename_component(FILENAME ${MODEL_FILE} NAME)
        add_custom_command(TARGET ${EXECUTABLE_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_FILE} $<TARGET_FILE_DIR:${EXECUTABLE_TARGET}>/models/${FILENAME}
            VERBATIM
        )
    endforeach()
   
    add_custom_target(${TARGET} DEPENDS ${SLANG_SPV_FILES})
endfunction()

project(shader-forge)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/render_passes
    ${VULKAN_INCLUDE_DIR}
)

set(IMGUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_demo.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_draw.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_tables.cpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_widgets.cpp  
    # Add your backend files too:
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_impl_vulkan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_impl_glfw.cpp  # or your window backend
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
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/render_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/scene_elements.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/node_gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/node_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/material_editor_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/scene_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/stb_implementations.cpp
    ${IMGUI_SOURCES}
)

target_precompile_headers(shader-forge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/modules/pch.hpp)

if(MSVC)
    target_compile_options(shader-forge PRIVATE /std:c++20 /Zc:__cplusplus /MP)
endif()

target_compile_definitions(shader-forge PRIVATE
    $<$<CONFIG:Debug>:DEBUG=1>
    $<$<CONFIG:Release>:DEBUG=0>
    # used by the runtime .slang hot-reload to locate the source shaders and the compiler
    SLANGC_PATH="${SLANGC_EXECUTABLE}"
    SHADER_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

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
cmake_minimum_required(VERSION 3.20)
project(VulkanTriangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch VMA (Vulkan Memory Allocator)
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(vma)

# Fetch GLM (math library)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8
)
FetchContent_MakeAvailable(glm)

# Main executable
add_executable(VulkanTriangle
    src/main.cpp
    src/Vulkan.cpp
    src/Camera.cpp
    src/GaussianData.cpp
    src/PlyLoader.cpp
)
target_link_libraries(VulkanTriangle PRIVATE Vulkan::Vulkan glfw GPUOpen::VulkanMemoryAllocator glm::glm)
target_include_directories(VulkanTriangle PRIVATE ${Vulkan_INCLUDE_DIRS})

# --- Shader compilation (HLSL → SPIR-V via DXC) ---
find_program(DXC dxc HINTS "$ENV{VULKAN_SDK}/Bin")

if(NOT DXC)
    message(FATAL_ERROR "dxc not found. Make sure the Vulkan SDK is installed and VULKAN_SDK is set.")
endif()

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

# Helper function: compile one HLSL shader to SPIR-V
function(compile_hlsl SOURCE PROFILE OUTPUT_NAME)
    set(SPIRV_OUTPUT ${SHADER_BIN_DIR}/${OUTPUT_NAME})
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${DXC} -spirv -fspv-target-env=vulkan1.3 -T ${PROFILE} -E main ${SOURCE} -Fo ${SPIRV_OUTPUT}
        DEPENDS ${SOURCE}
        COMMENT "Compiling ${OUTPUT_NAME} (HLSL -> SPIR-V)"
    )
    set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} ${SPIRV_OUTPUT} PARENT_SCOPE)
endfunction()

compile_hlsl(${SHADER_SRC_DIR}/triangle.vert.hlsl vs_6_0 triangle.vert.spv)
compile_hlsl(${SHADER_SRC_DIR}/triangle.frag.hlsl ps_6_0 triangle.frag.spv)
compile_hlsl(${SHADER_SRC_DIR}/points.vert.hlsl   vs_6_0 points.vert.spv)
compile_hlsl(${SHADER_SRC_DIR}/points.geom.hlsl   gs_6_0 points.geom.spv)
compile_hlsl(${SHADER_SRC_DIR}/points.frag.hlsl   ps_6_0 points.frag.spv)

add_custom_target(Shaders ALL DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(VulkanTriangle Shaders)

# Copy shaders next to the executable so the app can find them
add_custom_command(TARGET VulkanTriangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_BIN_DIR}
        $<TARGET_FILE_DIR:VulkanTriangle>/shaders
)

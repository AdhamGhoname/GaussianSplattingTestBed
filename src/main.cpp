#include "Vulkan.h"
#include "Camera.h"
#include "PlyLoader.h"
#include "GaussianData.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

constexpr uint32_t WIDTH  = 1280;
constexpr uint32_t HEIGHT = 720;

// ─── Application-specific types ─────────────────────────────────────────────

struct CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
};

struct GaussianPushConstants {
    VkDeviceAddress splatsAddr;
};

struct GaussianBuffers {
    AllocatedBuffer unified{};
    uint32_t        count = 0;
    VkDeviceAddress splatsAddr = 0;
};

// ─── Application GPU resources ──────────────────────────────────────────────

struct AppResources {
    VkDescriptorSetLayout          descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout               pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline                     graphicsPipeline    = VK_NULL_HANDLE;
    VkDescriptorPool               descriptorPool      = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>   descriptorSets;
    std::vector<AllocatedBuffer>   uniformBuffers;
    std::vector<void*>             uniformBuffersMapped;
    GaussianBuffers                gaussianBuffers{};
};

// ─── Application resource initialization ────────────────────────────────────

static void createDescriptorSetLayout(VulkanContext& vk, AppResources& app) {
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &uboBinding;

    if (vkCreateDescriptorSetLayout(vk.device(), &layoutInfo, nullptr, &app.descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout");
}

static void createGraphicsPipeline(VulkanContext& vk, AppResources& app) {
    auto vertCode = VulkanContext::readFile("shaders/points.vert.spv");
    auto fragCode = VulkanContext::readFile("shaders/points.frag.spv");

    VkShaderModule vertModule = vk.createShaderModule(vertCode);
    VkShaderModule fragModule = vk.createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset     = 0;
    pushConstantRange.size       = sizeof(GaussianPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 1;
    pipelineLayoutInfo.pSetLayouts            = &app.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges    = &pushConstantRange;
    if (vkCreatePipelineLayout(vk.device(), &pipelineLayoutInfo, nullptr, &app.pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = app.pipelineLayout;
    pipelineInfo.renderPass          = vk.renderPass();
    pipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(vk.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app.graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline");

    vkDestroyShaderModule(vk.device(), fragModule, nullptr);
    vkDestroyShaderModule(vk.device(), vertModule, nullptr);
}

static void createUniformBuffers(VulkanContext& vk, AppResources& app) {
    VkDeviceSize bufferSize = sizeof(CameraUBO);
    app.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    app.uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        app.uniformBuffers[i] = vk.createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        vmaMapMemory(vk.vmaAllocator(), app.uniformBuffers[i].allocation, &app.uniformBuffersMapped[i]);
    }
}

static void createDescriptorPool(VulkanContext& vk, AppResources& app) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    if (vkCreateDescriptorPool(vk.device(), &poolInfo, nullptr, &app.descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool");
}

static void createDescriptorSets(VulkanContext& vk, AppResources& app) {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, app.descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = app.descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts        = layouts.data();

    app.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(vk.device(), &allocInfo, app.descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets");

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = app.uniformBuffers[i].buffer;
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(CameraUBO);

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = app.descriptorSets[i];
        write.dstBinding      = 0;
        write.dstArrayElement = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &uboInfo;

        vkUpdateDescriptorSets(vk.device(), 1, &write, 0, nullptr);
    }
}

static GaussianBuffers uploadGaussianData(VulkanContext& vk, const GaussianGPUData& gpuData) {
    GaussianBuffers buffers;
    buffers.count = gpuData.count;
    if (gpuData.count == 0) return buffers;

    const VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    const VkDeviceSize totalSize = gpuData.splats.size() * sizeof(GaussianUnified);
    buffers.unified = vk.createDeviceLocalBuffer(gpuData.splats.data(), totalSize, ssboUsage);

    std::cout << "Uploaded " << gpuData.count << " Gaussian splats to GPU ("
              << totalSize / (1024 * 1024) << " MB)\n";

    buffers.splatsAddr = vk.getBufferDeviceAddress(buffers.unified.buffer);

    return buffers;
}

static void initAppResources(VulkanContext& vk, AppResources& app) {
    createDescriptorSetLayout(vk, app);
    createGraphicsPipeline(vk, app);
    createUniformBuffers(vk, app);
    createDescriptorPool(vk, app);
    createDescriptorSets(vk, app);
}

static void cleanupAppResources(VulkanContext& vk, AppResources& app) {
    if (app.gaussianBuffers.count > 0) {
        vk.destroyBuffer(app.gaussianBuffers.unified);
        app.gaussianBuffers.count = 0;
    }

    for (size_t i = 0; i < app.uniformBuffers.size(); i++) {
        vmaUnmapMemory(vk.vmaAllocator(), app.uniformBuffers[i].allocation);
        vk.destroyBuffer(app.uniformBuffers[i]);
    }
    app.uniformBuffers.clear();
    app.uniformBuffersMapped.clear();

    vkDestroyDescriptorPool(vk.device(), app.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(vk.device(), app.descriptorSetLayout, nullptr);
    vkDestroyPipeline(vk.device(), app.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(vk.device(), app.pipelineLayout, nullptr);
}

// ─── Input state passed through GLFW user pointer ───────────────────────────
struct InputState {
    Camera*        camera       = nullptr;
    VulkanContext* vk           = nullptr;
    double         lastX        = 0.0;
    double         lastY        = 0.0;
    bool           rightPressed = false;
    bool           firstMouse   = true;
    float          deltaTime    = 0.0f;
    float          lastFrame    = 0.0f;
};

// ─── GLFW callbacks ─────────────────────────────────────────────────────────

static void framebufferResizeCallback(GLFWwindow* window, int, int) {
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    input->vk->notifyFramebufferResized();
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        input->rightPressed = (action == GLFW_PRESS);
        if (input->rightPressed) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            input->firstMouse = true;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!input->rightPressed) return;

    if (input->firstMouse) {
        input->lastX = xpos;
        input->lastY = ypos;
        input->firstMouse = false;
        return;
    }

    float xOffset = static_cast<float>(xpos - input->lastX);
    float yOffset = static_cast<float>(input->lastY - ypos);  // reversed: y goes bottom-to-top
    input->lastX = xpos;
    input->lastY = ypos;

    input->camera->processMouseMovement(xOffset, yOffset);
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    input->camera->processMouseScroll(static_cast<float>(yoffset));
}

static void processKeyboardInput(GLFWwindow* window, Camera& camera, float deltaTime) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Forward, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Backward, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Left, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Right, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Up, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        camera.processKeyboard(CameraMovement::Down, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Gaussian Splatting", nullptr, nullptr);

        Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));

        VulkanContext vk;
        vk.init(window);

        // Create application-specific GPU resources
        AppResources app;
        initAppResources(vk, app);

        // Wire up input state
        InputState input;
        input.camera = &camera;
        input.vk     = &vk;
        glfwSetWindowUserPointer(window, &input);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetScrollCallback(window, scrollCallback);

        // Load .ply file if provided on the command line.
        if (argc > 1) {
            std::cout << "Loading PLY: " << argv[1] << "\n";
            GaussianCloud cloud = loadPlyFile(argv[1]);
            GaussianGPUData gpuData = prepareGPUData(cloud);
            app.gaussianBuffers = uploadGaussianData(vk, gpuData);
            std::cout << "Gaussians on GPU: " << app.gaussianBuffers.count << "\n";
        } else {
            std::cout << "No .ply file specified. Pass a path as the first argument.\n";
        }

        while (!glfwWindowShouldClose(window)) {
            float currentFrame = static_cast<float>(glfwGetTime());
            input.deltaTime = currentFrame - input.lastFrame;
            input.lastFrame = currentFrame;

            glfwPollEvents();
            processKeyboardInput(window, camera, input.deltaTime);

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            float aspect = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;

            CameraUBO camUBO{};
            camUBO.view = camera.getViewMatrix();
            camUBO.proj = camera.getProjectionMatrix(aspect);

            FrameInfo frame;
            if (!vk.beginFrame(frame))
                continue;

            // Update camera uniform buffer for this frame
            memcpy(app.uniformBuffersMapped[frame.frameIndex], &camUBO, sizeof(CameraUBO));

            // Record commands
            vk.beginRenderPass(frame.commandBuffer, frame.imageIndex);

            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.graphicsPipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    app.pipelineLayout, 0, 1,
                                    &app.descriptorSets[frame.frameIndex], 0, nullptr);

            if (app.gaussianBuffers.count > 0) {
                GaussianPushConstants pc{};
                pc.splatsAddr = app.gaussianBuffers.splatsAddr;
                vkCmdPushConstants(frame.commandBuffer, app.pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                vkCmdDraw(frame.commandBuffer, app.gaussianBuffers.count, 1, 0, 0);
            }

            vk.endRenderPass(frame.commandBuffer);
            vk.endFrame(frame);
        }

        vk.waitIdle();
        cleanupAppResources(vk, app);
        vk.cleanup();

        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

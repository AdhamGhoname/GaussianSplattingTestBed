#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// ─── constants ───────────────────────────────────────────────────────────────
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

// ─── helper structs ──────────────────────────────────────────────────────────
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// ─── VMA-backed buffer ──────────────────────────────────────────────────────
struct AllocatedBuffer {
    VkBuffer       buffer     = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
};

// ─── Per-frame info returned by beginFrame() ────────────────────────────────
struct FrameInfo {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint32_t        imageIndex    = 0;
    uint32_t        frameIndex    = 0;
};

// ═════════════════════════════════════════════════════════════════════════════
//  VulkanContext — generic Vulkan abstraction
// ═════════════════════════════════════════════════════════════════════════════
class VulkanContext {
public:
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VulkanContext()  = default;
    ~VulkanContext() = default;

    /// Call once after GLFW window creation.
    void init(GLFWwindow* window);

    /// Release core Vulkan resources. App-owned resources must be
    /// destroyed before calling this. Safe to call multiple times.
    void cleanup();

    /// Notify the context that the framebuffer was resized.
    void notifyFramebufferResized() { framebufferResized_ = true; }

    /// Wait for the device to become idle.
    void waitIdle();

    // ── Frame management ─────────────────────────────────────────────────

    /// Acquire next image, wait for fences, begin command buffer.
    /// Returns false if the swap chain was recreated (skip this frame).
    bool beginFrame(FrameInfo& frame);

    /// Begin the default render pass for the acquired image.
    void beginRenderPass(VkCommandBuffer cmd, uint32_t imageIndex);

    /// End the default render pass.
    void endRenderPass(VkCommandBuffer cmd);

    /// End command buffer recording, submit, and present.
    void endFrame(const FrameInfo& frame);

    // ── Buffer helpers (VMA) ─────────────────────────────────────────────
    AllocatedBuffer createBuffer(VkDeviceSize size,
                                 VkBufferUsageFlags usage,
                                 VmaMemoryUsage memoryUsage) const;

    void uploadToBuffer(const AllocatedBuffer& buffer,
                        const void* src,
                        VkDeviceSize size) const;

    void destroyBuffer(AllocatedBuffer& buffer) const;

    AllocatedBuffer createDeviceLocalBuffer(const void* data,
                                            VkDeviceSize size,
                                            VkBufferUsageFlags usage) const;

    // ── Shader helpers ───────────────────────────────────────────────────
    VkShaderModule createShaderModule(const std::vector<char>& code);

    // ── Buffer device address ────────────────────────────────────────────
    VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer) const;

    // ── Utilities ────────────────────────────────────────────────────────
    static std::vector<char> readFile(const std::string& filename);

    // ── Accessors ────────────────────────────────────────────────────────
    VkDevice         device()          const { return device_; }
    VkPhysicalDevice physicalDevice()  const { return physicalDevice_; }
    VmaAllocator     vmaAllocator()    const { return allocator_; }
    VkRenderPass     renderPass()      const { return renderPass_; }
    VkExtent2D       swapChainExtent() const { return swapChainExtent_; }
    VkCommandPool    commandPool()     const { return commandPool_; }
    VkQueue          graphicsQueue()   const { return graphicsQueue_; }

private:
    GLFWwindow* window_ = nullptr;

    // ── core objects ─────────────────────────────────────────────────────
    VkInstance                 instance_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT  debugMessenger_  = VK_NULL_HANDLE;
    VkSurfaceKHR              surface_         = VK_NULL_HANDLE;
    VkPhysicalDevice          physicalDevice_  = VK_NULL_HANDLE;
    VkDevice                  device_          = VK_NULL_HANDLE;
    VkQueue                   graphicsQueue_   = VK_NULL_HANDLE;
    VkQueue                   presentQueue_    = VK_NULL_HANDLE;
    VmaAllocator              allocator_       = VK_NULL_HANDLE;

    // ── swap chain ───────────────────────────────────────────────────────
    VkSwapchainKHR             swapChain_       = VK_NULL_HANDLE;
    std::vector<VkImage>       swapChainImages_;
    VkFormat                   swapChainImageFormat_{};
    VkExtent2D                 swapChainExtent_{};
    std::vector<VkImageView>   swapChainImageViews_;
    std::vector<VkFramebuffer> swapChainFramebuffers_;

    // ── depth buffer ─────────────────────────────────────────────────────
    VkImage        depthImage_       = VK_NULL_HANDLE;
    VmaAllocation  depthAllocation_  = VK_NULL_HANDLE;
    VkImageView    depthImageView_   = VK_NULL_HANDLE;

    // ── render pass ──────────────────────────────────────────────────────
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    // ── commands ─────────────────────────────────────────────────────────
    VkCommandPool                commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    // ── sync ─────────────────────────────────────────────────────────────
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t                 currentFrame_       = 0;
    bool                     framebufferResized_ = false;

    // ── private helpers ──────────────────────────────────────────────────
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();

    void cleanupSwapChain();
    void recreateSwapChain();

    // ── queries ──────────────────────────────────────────────────────────
    bool                    checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    bool                    isDeviceSuitable(VkPhysicalDevice dev);
    bool                    checkDeviceExtensionSupport(VkPhysicalDevice dev);
    QueueFamilyIndices      findQueueFamilies(VkPhysicalDevice dev);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev);
    VkSurfaceFormatKHR      chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available);
    VkPresentModeKHR        chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available);
    VkExtent2D              chooseSwapExtent(const VkSurfaceCapabilitiesKHR& cap);

    static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& ci);
};

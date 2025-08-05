#include <core.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

class App {

  public:
    App() = default;

    void run() {
        initWindow();
        renderer.setWindow(window);
        renderer.initVulkan();
        renderer.initializeResourceDefaults();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    Renderer renderer;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);
    }

    static void framebufferResizedCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<App*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            renderer.drawFrame();
        }
        renderer.getDevices()->getLogicalDevice().waitIdle();
    }

    void cleanup() {

        renderer.getDevices()->getLogicalDevice().waitIdle();
        renderer.cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main(){
    App app;
    app.run();
}

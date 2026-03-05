#include "Camera.h"

#include <algorithm>
#include <cmath>

Camera::Camera(glm::vec3 position, float yaw, float pitch)
    : position_(position), yaw_(yaw), pitch_(pitch)
{
    updateVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
    auto proj = glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
    //proj[1][1] *= -1.0f;   // Vulkan Y-axis is flipped vs OpenGL
    return proj;
}

void Camera::processKeyboard(CameraMovement direction, float deltaTime) {
    float velocity = speed * deltaTime;
    switch (direction) {
        case CameraMovement::Forward:  position_ += front_ * velocity; break;
        case CameraMovement::Backward: position_ -= front_ * velocity; break;
        case CameraMovement::Left:     position_ -= right_ * velocity; break;
        case CameraMovement::Right:    position_ += right_ * velocity; break;
        case CameraMovement::Up:       position_ += worldUp_ * velocity; break;
        case CameraMovement::Down:     position_ -= worldUp_ * velocity; break;
    }
}

void Camera::processMouseMovement(float xOffset, float yOffset) {
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    yaw_   += xOffset;
    pitch_ -= yOffset;

    pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
    updateVectors();
}

void Camera::processMouseScroll(float yOffset) {
    fov -= yOffset;
    fov = std::clamp(fov, 1.0f, 120.0f);
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front_ = glm::normalize(front);
    right_ = glm::normalize(glm::cross(front_, worldUp_));
    up_    = glm::normalize(glm::cross(right_, front_));
}

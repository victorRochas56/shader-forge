#pragma once
#include "scene_elements.hpp"
#include "gizmo.hpp"
#include "utils.hpp"

namespace Manip {

    enum ManipShapeType {
        Circle,
        Arrow,
        Plane,
    };

    struct Shape {
        ManipShapeType shape;
        glm::vec3 position;
        float radius;
        glm::vec3 axis;

        bool cast(glm::vec3& origin, glm::vec3& rayDir) {
            switch (shape)
            {
            case ManipShapeType::Circle:

                break;
            
            case ManipShapeType::Arrow:
                
                break;

            case ManipShapeType::Plane:
                
                break;

            default:
                break;
            }
        }
    };

    std::vector<Shape> manipShapes; 

    void showManipulator(Node& node, Camera& camera, const glm::vec2& ndcMousePos) {
        glm::vec3 origin;
        glm::vec3 direction;
        camera.rayFromScreenCoords(ndcMousePos.x,ndcMousePos.y,origin,direction);
        glm::vec3 worldPos = node.getWorldPosition(); 

        float tX = 0;
        glm::vec4 xCol(1,0,0,1);
        float tY = 0;
        glm::vec4 yCol(0,1,0,1);
        float tZ = 0;
        glm::vec4 zCol(0,0,1,1);
        intersectCircle(node.right(),worldPos,0.2f,0.02f,origin, direction,tX);
        intersectCircle(node.up(),worldPos,0.2f,0.02f,origin, direction,tY);
        intersectCircle(node.forward(),worldPos,0.2f,0.02f,origin, direction,tZ);

        if(tX != 0 && (tY == 0 || tX < tY) && (tZ == 0 || tX < tZ))
            xCol = {1,1,1,1};
        if(tY != 0 && (tX == 0 ||tY < tX) && (tZ == 0 || tY < tZ))
            yCol = {1,1,1,1};
        if(tZ != 0 && (tX == 0 || tZ < tX) && (tY == 0 || tZ < tY))
            zCol = {1,1,1,1};
        Gizmos::drawCircle(node.getWorldPosition(),0.2f,node.right(),xCol);
        Gizmos::drawCircle(node.getWorldPosition(),0.2f,node.up(),yCol);
        Gizmos::drawCircle(node.getWorldPosition(),0.2f,node.forward(),zCol);
        return;
    }

    void castManipulators(glm::vec3 origin, glm::vec3 dir) {
    }

}
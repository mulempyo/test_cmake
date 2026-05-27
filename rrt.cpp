#include "rrt_star_planner/rrt_star.h"
#include "pluginlib/class_list_macros.hpp"
#include <nav_msgs/msg/path.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <mutex>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

PLUGINLIB_EXPORT_CLASS(rrt_star::RRTStarPlanner, nav2_core::GlobalPlanner)

namespace rrt_star {

RRTStarPlanner::RRTStarPlanner() : initialized_(false), goal_threshold_(0.5), step_size_(0.05), max_iterations_(100000), rewire_radius_(0.05) {}

void RRTStarPlanner::configure(
         const rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
         std::string name,
         std::shared_ptr<tf2_ros::Buffer> tf,
         std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) {
  if(!initialized_){
  node_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  origin_x_ = costmap_->getOriginX();
  origin_y_ = costmap_->getOriginY();
  resolution_ = costmap_->getResolution();
  width_ = costmap_->getSizeInCellsX();
  height_ = costmap_->getSizeInCellsY();
  global_frame_ = costmap_ros_->getGlobalFrameID();
  checker_ = std::make_shared<nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D*>>(costmap_);

  vis_path_pub_ =
  node_->create_publisher<nav_msgs::msg::Path>(
  "rrt_star_path",
  rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  tree_pub_ =
  node_->create_publisher<visualization_msgs::msg::Marker>(
  "rrt_star_tree", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
  initialized_ = true;
  }
  
}

void RRTStarPlanner::cleanup() {
    vis_path_pub_.reset();
    tree_pub_.reset();
  RCLCPP_INFO(
    node_->get_logger(), "CleaningUp plugin %s of type RRTStarPlanner",
    name_.c_str());
}

void RRTStarPlanner::activate() {
    vis_path_pub_->on_activate();
    tree_pub_->on_activate();
  RCLCPP_INFO(
    node_->get_logger(), "Activating plugin %s of type RRTStarPlanner",
    name_.c_str());
}

void RRTStarPlanner::deactivate() {
    vis_path_pub_->on_deactivate();
    tree_pub_->on_deactivate();
  RCLCPP_INFO(
    node_->get_logger(), "Deactivating plugin %s of type RRTStarPlanner",
    name_.c_str());
}

nav_msgs::msg::Path RRTStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped& start,
  const geometry_msgs::msg::PoseStamped& goal) {
  boost::mutex::scoped_lock lock(mutex_);

  plan.clear();
  tree.clear();
  costs_.clear();

  return_path.header.frame_id = costmap_ros_->getGlobalFrameID();
  return_path.header.stamp = node_->now();

  tf2::Quaternion quat;
  tf2::fromMsg(goal.pose.orientation, quat);

  double goal_yaw, unused_pitch, unused_roll;
  tf2::Matrix3x3(quat).getEulerYPR(goal_yaw, unused_pitch, unused_roll);

  if (goal.header.frame_id != costmap_ros_->getGlobalFrameID()) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "The RRT planner can only accept goals in the %s frame, but a goal was sent in the %s frame.",
      costmap_ros_->getGlobalFrameID().c_str(), goal.header.frame_id.c_str());
    return return_path;
  }

  unsigned int start_x, start_y, goal_x, goal_y;
  if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y)) {
    RCLCPP_WARN(node_->get_logger(), "The start is out of the map bounds.");
    return return_path;
  }

  if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
    RCLCPP_WARN(node_->get_logger(), "The goal is out of the map bounds.");
    return return_path;
  }

  unsigned int start_index = start_y * width_ + start_x;
  unsigned int goal_index = goal_y * width_ + goal_x;

  tree.emplace_back(start_index, start_index);
  costs_[start_index] = 0.0;

  for (int i = 0; i < max_iterations_; ++i) {
        double random_x, random_y, random_th;
        createRandomValidPose(random_x, random_y, random_th);

        unsigned int nearest_index = nearestNode(random_x, random_y);
        if (nearest_index == std::numeric_limits<unsigned int>::max()) {
            continue;
        }

        double nearest_x, nearest_y;
        costmap_->mapToWorld(nearest_index % width_, nearest_index / width_, nearest_x, nearest_y);

        double new_x, new_y, new_th;
        createPoseWithinRange(nearest_x, nearest_y, 0.0, random_x, random_y, random_th, step_size_, new_x, new_y, new_th);

        if (isValidPathBetweenPoses(nearest_x, nearest_y, 0.0, new_x, new_y, 0.0)) {
            unsigned int new_x_int, new_y_int, new_index, cost;

            if(isValidPose(new_x,new_y)){
             costmap_->worldToMap(new_x, new_y, new_x_int, new_y_int);
             cost = costmap_->getCost(new_x_int, new_y_int);
             new_index = new_y_int * width_ + new_x_int;
            }

            if (costs_.find(new_index) == costs_.end()) {
                tree.emplace_back(new_index, nearest_index);
                costs_[new_index] = costs_[nearest_index] + distance(nearest_x, nearest_y, new_x, new_y);

                rewire(new_index);

                if (isValidPathBetweenPoses(new_x, new_y, 0.0, goal.pose.position.x, goal.pose.position.y, goal_yaw)) {
                    tree.emplace_back(goal_index, new_index);
                    costs_[goal_index] = costs_[new_index] + distance(new_x, new_y, goal.pose.position.x, goal.pose.position.y);
                    break;
                }
            }
        }
     visualizeTree();
    }

    return_path = constructPath(start_index, goal_index, plan);
    vis_path_pub_->publish(return_path);
    return return_path;
}

void RRTStarPlanner::rewire(unsigned int& new_index) {
    double new_x, new_y;
    costmap_->mapToWorld(new_index % width_, new_index / width_, new_x, new_y);

    unsigned int closest_neighbor = new_index;
    double min_cost = std::numeric_limits<double>::max();

    for (const auto& node : tree) {
        unsigned int neighbor_index = node.first;
        if (neighbor_index == new_index) continue;

        double neighbor_x, neighbor_y;
        costmap_->mapToWorld(neighbor_index % width_, neighbor_index / width_, neighbor_x, neighbor_y);

        if (distance(new_x, new_y, neighbor_x, neighbor_y) <= rewire_radius_ && isValidPose(new_x, new_y) && isValidPose(neighbor_x, neighbor_y)
            && isWithinMapBounds(new_x, new_y) && isWithinMapBounds(neighbor_x, neighbor_y)) {
            double potential_cost = costs_[new_index] + distance(new_x, new_y, neighbor_x, neighbor_y);

            double obstacle_cost = footprintCost(neighbor_x, neighbor_y, 0.0);
            potential_cost += obstacle_cost * 1.0; 

            if (potential_cost < costs_[neighbor_index] && isValidPathBetweenPoses(new_x, new_y, 0.0, neighbor_x, neighbor_y, 0.0)) {
                costs_[neighbor_index] = potential_cost;

                auto it = std::find_if(tree.begin(), tree.end(),
                    [neighbor_index](const std::pair<unsigned int, unsigned int>& node) {
                        return node.first == neighbor_index;
                    });

                if (it != tree.end()) {
                    it->second = new_index;
                }
            }
        }
    }

}

nav_msgs::msg::Path RRTStarPlanner::constructPath(unsigned int start_index, unsigned int goal_index, std::vector<geometry_msgs::msg::PoseStamped>& plan) {
    if (costs_.find(goal_index) == costs_.end()) {
        RCLCPP_WARN(node_->get_logger(),"No valid path found to goal.");
    }

    unsigned int current_index = goal_index;
    while (current_index != start_index) {
        double wx, wy;
        costmap_->mapToWorld(current_index % width_, current_index / width_, wx, wy);

        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = costmap_ros_->getGlobalFrameID();
        pose.header.stamp = node_->now();
        pose.pose.position.x = wx;
        pose.pose.position.y = wy;
        pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, 0, 1));
        plan.push_back(pose);

        auto it = std::find_if(tree.begin(), tree.end(), [current_index](const std::pair<unsigned int, unsigned int>& node) {
            return node.first == current_index;
        });

        if (it == tree.end()) {
            RCLCPP_WARN(node_->get_logger(),"Failed to backtrack path.");
        }

        current_index = it->second;
    }

    std::reverse(plan.begin(), plan.end());
    plan = smoothPath(plan);
    
    nav_msgs::msg::Path construct_path;
    construct_path.header.frame_id = costmap_ros_->getGlobalFrameID();
    construct_path.header.stamp = node_->now();

    construct_path = publishPlan(plan);
    return construct_path;
}

double RRTStarPlanner::footprintCost(double x, double y, double th) const {
  if (!initialized_) {
    RCLCPP_WARN(node_->get_logger(),"The RRT Planner has not been initialized, you must call initialize().");
    return -1.0;
  }

  std::vector<geometry_msgs::msg::Point> footprint = costmap_ros_->getRobotFootprint();

  if (footprint.size() < 3) return -1.0;

  double footprint_cost = checker_->footprintCostAtPose(x, y, th, footprint);

  return footprint_cost;
}

bool RRTStarPlanner::isValidPose(double x, double y, double th) const {
  double footprint_cost = footprintCost(x, y, th);

  if ((footprint_cost < 0) || (footprint_cost > 128)) {
    return false;
  }

  return true;
}

bool RRTStarPlanner::isValidPose(double x, double y) const {
  unsigned int mx, my, cost = 0;
  if (costmap_->worldToMap(x, y, mx, my)) {
    for (int dx = -3; dx <= 3; ++dx) {
      for (int dy = -3; dy <= 3; ++dy) {
        unsigned int nx = mx + dx;
        unsigned int ny = my + dy;
        if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
          cost = costmap_->getCost(nx, ny);
      
        if (cost > 128) {
          return false;
        }
      }
    }
  }
  if(cost == nav2_costmap_2d::FREE_SPACE){
    return true;
  }
}

void RRTStarPlanner::createRandomValidPose(double &x, double &y, double &th) const {
  double wx_min, wy_min;
  costmap_->mapToWorld(0, 0, wx_min, wy_min);

  double wx_max, wy_max;
  unsigned int mx_max = costmap_->getSizeInCellsX();
  unsigned int my_max = costmap_->getSizeInCellsY();
  costmap_->mapToWorld(mx_max, my_max, wx_max, wy_max);

  bool found_pose = false;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  int max_attempts = 500;  
  int attempts = 0;

  while (!found_pose && attempts < max_attempts) {
        double wx_rand = wx_min + dis(gen) * (wx_max - wx_min);
        double wy_rand = wy_min + dis(gen) * (wy_max - wy_min);
        double th_rand = -M_PI + dis(gen) * (2.0 * M_PI);

        if (isValidPose(wx_rand, wy_rand, th_rand) && isValidPose(wx_rand, wy_rand)) {
            x = wx_rand;
            y = wy_rand;
            th = th_rand;
            found_pose = true;
        }

        attempts++;
    }

    if (!found_pose) {
        RCLCPP_WARN(node_->get_logger(), "Failed to find a valid pose after %d attempts. Returning last sample.", max_attempts);
    }

}

unsigned int RRTStarPlanner::nearestNode(double random_x, double random_y) {
  unsigned int global_nearest_index = 0;
    double global_min_dist = std::numeric_limits<double>::max();

    std::vector<std::pair<unsigned int, double>> candidates;

    for (const auto& node : tree) {
        unsigned int node_index = node.first;

        double node_x, node_y;
        costmap_->mapToWorld(node_index % width_, node_index / width_, node_x, node_y);

        double dist = distance(node_x, node_y, random_x, random_y);

        if (dist < global_min_dist && dist > 0.001) {
            if (isValidPose(node_x, node_y) && isWithinMapBounds(node_x, node_y)) {
                global_min_dist = dist;
                global_nearest_index = node_index;
            }
        }

        if (dist <= rewire_radius_ && dist > 0.001) {
            
            if (isValidPose(node_x, node_y) && isValidPose(random_x, random_y) &&
                isWithinMapBounds(node_x, node_y) && isWithinMapBounds(random_x, random_y)) {
                
                candidates.emplace_back(node_index, dist);
            }
        }
    }

    if (candidates.empty()) {
        return global_nearest_index;
    }

    unsigned int best_index = 0;
    double best_cost = std::numeric_limits<double>::max();
    double best_dist = std::numeric_limits<double>::max();

    for (const auto& c : candidates) {
        unsigned int cand_index = c.first;
        double cand_dist = c.second;

        double cand_cost = costs_[cand_index];

        if ((cand_cost < best_cost) ||
            (std::fabs(cand_cost - best_cost) < 1e-9 && cand_dist < best_dist))
        {
            best_cost = cand_cost;
            best_dist = cand_dist;
            best_index = cand_index;
        }
    }

    return best_index;
}

void RRTStarPlanner::createPoseWithinRange(double start_x, double start_y, double start_th,
                                       double end_x, double end_y, double end_th,
                                       double range, double &new_x, double &new_y, double &new_th) const { //world

  double x_step = end_x - start_x;
  double y_step = end_y - start_y;
  double mag = sqrt((x_step * x_step) + (y_step * y_step));

  double newX, newY, newTh;

  if (mag < 0.001) {
    new_x = end_x;
    new_y = end_y;
    new_th = end_th;
    return;
  }

  x_step /= mag;
  y_step /= mag;

  newX = start_x + x_step * range;
  newY = start_y + y_step * range;
  newTh = start_th;

  if(isValidPose(newX, newY)){
    new_x = newX;
    new_y = newY;
    new_th = newTh;
  }
}

bool RRTStarPlanner::isValidPathBetweenPoses(double x1, double y1, double th1,
                                         double x2, double y2, double th2) const {
  double interp_step_size = 0.05; 
  double current_step = interp_step_size;
  double d = std::hypot(x2 - x1, y2 - y1);

  while (current_step < d) {
    double interp_x, interp_y, interp_th;
    createPoseWithinRange(x1, y1, th1, x2, y2, th2, current_step, interp_x, interp_y, interp_th);

    if (!isValidPose(interp_x, interp_y, interp_th)) {
      return false;
    }

    current_step += interp_step_size;
  }

  return true;
}

bool RRTStarPlanner::isWithinMapBounds(double x, double y) const {
    unsigned int mx, my;
    if (!costmap_->worldToMap(x, y, mx, my)) {
        return false;
    }
    return true;
}

void RRTStarPlanner::visualizeTree() const {
  if (!initialized_) {
    RCLCPP_WARN(node_->get_logger(), "RRTStarPlanner not initialized");
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.lifetime = rclcpp::Duration::from_seconds(10.0); 
  marker.frame_locked = true;
  marker.header.frame_id = costmap_ros_->getGlobalFrameID();
  marker.header.stamp =  node_->now();
  marker.ns = "rrt_tree";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.02;
  marker.color.r = 0.0;
  marker.color.g = 0.8;
  marker.color.b = 0.2;
  marker.color.a = 1.0;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0; 
  marker.pose.orientation.w = 1.0;

  for (const auto& edge : tree) {
    unsigned int parent_index = edge.second;
    unsigned int child_index = edge.first;

    double parent_x, parent_y, child_x, child_y;
    costmap_->mapToWorld(parent_index % width_, parent_index / width_, parent_x, parent_y);
    costmap_->mapToWorld(child_index % width_, child_index / width_, child_x, child_y);

    if (isWithinMapBounds(parent_x, parent_y) && isWithinMapBounds(child_x, child_y)) {
      geometry_msgs::msg::Point p1, p2;
      p1.x = parent_x;
      p1.y = parent_y;
      p1.z = 0.0;
      p2.x = child_x;
      p2.y = child_y;
      p2.z = 0.0;
      marker.points.push_back(p1);
      marker.points.push_back(p2);
    }
  }

  tree_pub_->publish(marker);
}


nav_msgs::msg::Path RRTStarPlanner::publishPlan(const std::vector<geometry_msgs::msg::PoseStamped> &path) const {

  nav_msgs::msg::Path path_msg;
  path_msg.poses.resize(path.size());

  if (path.empty()) {
    path_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
    path_msg.header.stamp = node_->now();
  } else {
    path_msg.header.frame_id = path[0].header.frame_id;
    path_msg.header.stamp = path[0].header.stamp;
  }

  for (unsigned int i = 0; i < path.size(); i++) {
    path_msg.poses[i] = path[i];
  }
  return path_msg;
}

std::vector<geometry_msgs::msg::PoseStamped> RRTStarPlanner::smoothPath(
    const std::vector<geometry_msgs::msg::PoseStamped>& path_in) const 
{
  if (path_in.size() <= 2) {
    return path_in; 
  }

  std::vector<geometry_msgs::msg::PoseStamped> path_out;
  path_out.push_back(path_in.front());

  size_t i = 0;
  while (i < path_in.size() - 1) {
    size_t j = path_in.size() - 1;

    while (j > i + 1) {
      double x1 = path_in[i].pose.position.x;
      double y1 = path_in[i].pose.position.y;
      double x2 = path_in[j].pose.position.x;
      double y2 = path_in[j].pose.position.y;

      if (isValidPathBetweenPoses(x1, y1, 0.0, x2, y2, 0.0)) {
        break; 
      }
      --j;
    }

    path_out.push_back(path_in[j]);
    i = j;
  }

  return path_out;
}


double RRTStarPlanner::distance(double x1, double y1, double x2, double y2) {
  return std::hypot(x2 - x1, y2 - y1);
}

void RRTStarPlanner::mapToWorld(unsigned int mx, unsigned int my, double &wx, double &wy) {
  wx = origin_x_ + mx * resolution_;
  wy = origin_y_ + my * resolution_;
}

} // namespace rrt_star

    
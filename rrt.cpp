#include "rrt_planner/rrt.h"
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

PLUGINLIB_EXPORT_CLASS(rrt::RRTPlanner, nav2_core::GlobalPlanner)

namespace rrt {

RRTPlanner::RRTPlanner() : initialized_(false), goal_threshold_(0.5), step_size_(0.05) {}

void RRTPlanner::configure(
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
  "rrt_vis_path",
  rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  tree_pub_ =
  node_->create_publisher<visualization_msgs::msg::Marker>(
  "rrt_tree", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
  initialized_ = true;
  }
  
}

void RRTPlanner::cleanup() {
    vis_path_pub_.reset();
    tree_pub_.reset();
  RCLCPP_INFO(
    node_->get_logger(), "CleaningUp plugin %s of type RRTPlanner",
    name_.c_str());
}

void RRTPlanner::activate() {
    vis_path_pub_->on_activate();
    tree_pub_->on_activate();
  RCLCPP_INFO(
    node_->get_logger(), "Activating plugin %s of type RRTPlanner",
    name_.c_str());
}

void RRTPlanner::deactivate() {
    vis_path_pub_->on_deactivate();
    tree_pub_->on_deactivate();
  RCLCPP_INFO(
    node_->get_logger(), "Deactivating plugin %s of type RRTPlanner",
    name_.c_str());
}

nav_msgs::msg::Path RRTPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped& start,
  const geometry_msgs::msg::PoseStamped& goal) {
  boost::mutex::scoped_lock lock(mutex_);

  plan.clear();
  tree.clear();

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
  unsigned int final_node_index = 0;

  int i = 0;
  int max_iterations_ = 100000000;

  while (i < max_iterations_) {
    double random_x, random_y, random_th;
    createRandomValidPose(random_x, random_y, random_th);
    unsigned int nearest_index = nearestNode(random_x, random_y);
    RCLCPP_WARN(node_->get_logger(), "nearest_index: %f", nearest_index);
    if (nearest_index == -1) {
      RCLCPP_WARN(node_->get_logger(), "no valid nearest node found");
      return return_path;
    }

    double nearest_x, nearest_y;
    costmap_->mapToWorld(nearest_index % width_, nearest_index / width_, nearest_x, nearest_y);

    double th, new_x, new_y, new_th;
    createPoseWithinRange(nearest_x, nearest_y, th,
                          random_x, random_y, random_th, step_size_,
                          new_x, new_y, new_th);

    if (isValidPathBetweenPoses(nearest_x, nearest_y, th, new_x, new_y, new_th)) {
      unsigned int new_x_int, new_y_int, new_index;
      costmap_->worldToMap(new_x, new_y, new_x_int, new_y_int);
      new_index = new_y_int * width_ + new_x_int;
      RCLCPP_WARN(node_->get_logger(), "new index:%f", new_index);
      if (new_index != nearest_index && std::find_if(tree.begin(), tree.end(),
          [new_index](const std::pair<unsigned int, unsigned int>& node) {
            return node.first == new_index;
          }) == tree.end()) {
        tree.emplace_back(new_index, nearest_index);
      } else {
        RCLCPP_WARN(node_->get_logger(), "skipping invalid or duplicate node: %d", new_index);
        continue;
      }

      if (isValidPathBetweenPoses(new_x, new_y, 0, goal.pose.position.x, goal.pose.position.y, goal_yaw)) {
        final_node_index = goal_index;
        tree.emplace_back(goal_index, new_index);
        break; 
      }
    }
    i++;
    visualizeTree();
  }

  RCLCPP_WARN(node_->get_logger(), "tree_size: %d", tree.size());

  if (final_node_index == 0) {
    RCLCPP_WARN(node_->get_logger(), "final_node_index == 0, failed to find a valid path");
    return return_path;
  }

  if (final_node_index != 0) {
  unsigned int current_index = final_node_index;
  double wx, wy;
  unsigned int mx, my;
  mx = current_index % width_;
  my = current_index / width_;
  costmap_->mapToWorld(mx, my, wx, wy);

  while (current_index != start_index) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = costmap_ros_->getGlobalFrameID();
    pose.header.stamp = node_->get_clock()->now();
    pose.pose.position.x = wx;
    pose.pose.position.y = wy;
    pose.pose.position.z = 0;
    pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, 0, 1));
    plan.push_back(pose);

    auto it = std::find_if(tree.begin(), tree.end(),
      [current_index](const std::pair<unsigned int, unsigned int>& node) {
        return node.first == current_index;
      });

    if (it == tree.end()) {
      RCLCPP_WARN(node_->get_logger(), "failed to find next node for current_index:%d", current_index);
      break;
    }
    if (it->first == it->second) {
      RCLCPP_WARN(node_->get_logger(), "cycle detected in tree at index:%d", it->first);
      break;
    }

    RCLCPP_INFO(node_->get_logger(), "current_index:%d, next_index:%d", current_index, it->second);

    current_index = it->second;
    mx = current_index % width_;
    my = current_index / width_;
    costmap_->mapToWorld(mx, my, wx, wy);
  }

  std::reverse(plan.begin(), plan.end());
  RCLCPP_WARN(node_->get_logger(), "plan size: %d", plan.size());
  plan = smoothPath(plan);
 
  return_path = publishPlan(plan);
  vis_path_pub_->publish(return_path);
  return return_path;
  }
}

double RRTPlanner::footprintCost(double x, double y, double th) const {
  if (!initialized_) {
    RCLCPP_WARN(node_->get_logger(),"The RRT Planner has not been initialized, you must call initialize().");
    return -1.0;
  }

  std::vector<geometry_msgs::msg::Point> footprint = costmap_ros_->getRobotFootprint();

  if (footprint.size() < 3) return -1.0;

  double footprint_cost = checker_->footprintCostAtPose(x, y, th, footprint);

  return footprint_cost;
}

bool RRTPlanner::isValidPose(double x, double y, double th) const {
  double footprint_cost = footprintCost(x, y, th);

  if ((footprint_cost < 0) || (footprint_cost > 128)) {
    return false;
  }

  return true;
}

bool RRTPlanner::isValidPose(double x, double y) const {
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

void RRTPlanner::createRandomValidPose(double &x, double &y, double &th) const {
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

unsigned int RRTPlanner::nearestNode(double random_x, double random_y) {
  unsigned int nearest_index = 0;
  double min_dist = std::numeric_limits<double>::max();

  for (const auto &node : tree) {
    unsigned int node_index = node.first;
    double node_x, node_y;
    costmap_->mapToWorld(node_index % width_, node_index / width_, node_x, node_y);
    double dist = distance(node_x, node_y, random_x, random_y);

    if (dist < min_dist && dist > 0.001){
        if(isValidPose(node_x, node_y) && isWithinMapBounds(node_x, node_y)) {
        min_dist = dist;
        nearest_index = node_index;
      }
    }
  }

  return nearest_index;
}

void RRTPlanner::createPoseWithinRange(double start_x, double start_y, double start_th,
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

bool RRTPlanner::isValidPathBetweenPoses(double x1, double y1, double th1,
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

bool RRTPlanner::isWithinMapBounds(double x, double y) const {
    unsigned int mx, my;
    if (!costmap_->worldToMap(x, y, mx, my)) {
        return false;
    }
    return true;
}

void RRTPlanner::visualizeTree() const {
  if (!initialized_) {
    RCLCPP_WARN(node_->get_logger(), "RRTPlanner not initialized");
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


nav_msgs::msg::Path RRTPlanner::publishPlan(const std::vector<geometry_msgs::msg::PoseStamped> &path) const {

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

std::vector<geometry_msgs::msg::PoseStamped> RRTPlanner::smoothPath(
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


double RRTPlanner::distance(double x1, double y1, double x2, double y2) {
  return std::hypot(x2 - x1, y2 - y1);
}

void RRTPlanner::mapToWorld(unsigned int mx, unsigned int my, double &wx, double &wy) {
  wx = origin_x_ + mx * resolution_;
  wy = origin_y_ + my * resolution_;
}

}; // namespace rrt

    
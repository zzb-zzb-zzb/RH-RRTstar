#pragma once

#include <nav_msgs/OccupancyGrid.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/config.h>

#include <ros/ros.h>
#include <ros/console.h>

namespace ob = ompl::base;

class CollisionChecker
{
public:
  CollisionChecker(nav_msgs::OccupancyGridPtr ogm_map) : map_{ogm_map}
  {
    if (!getBounds(min_bound_x_, max_bound_x_, min_bound_y_, max_bound_y_, *map_))
    {
      std::cerr << "Fail to generate bounds in the occupancy grid map." << std::endl;
      exit(1);
    }

    map_cell_size_x_ = map_->info.width;
    map_cell_size_y_ = map_->info.height;
    map_resolution_ = map_->info.resolution;
    map_origin_x_ = map_->info.origin.position.x;
    map_origin_y_ = map_->info.origin.position.y;
  }

  bool isValid(const ob::State *state)
  {
    if (!state)
    {
      throw std::runtime_error("No state found for vertex");
    }

    // Convert to RealVectorStateSpace
    //  const ob::RealVectorStateSpace::StateType* real_state =
    //      static_cast<const ob::RealVectorStateSpace::StateType*>(state);
    //  double x = real_state->values[0];
    //  double y = real_state->values[1];
    //  Convert to SE2StateSpace
    const ob::SE2StateSpace::StateType *real_state =
        static_cast<const ob::SE2StateSpace::StateType *>(state);
    double x = real_state->getX();
    double y = real_state->getY();
    double yaw = real_state->getYaw();

    int N = 3;
    double l = 4.7;
    double lr = 0.9;
    // double l = 0.57;
    // double lr = 0.3;
    // double w=2;
    // double R = sqrt(pow(l/(2*N), 2) + pow(w/2, 2));


    double x_check = 0;
    double y_check = 0;

    for (int i = 0; i < N; ++i)
    {
      x_check = x + ((2 * i - 1) * l / (2 * N) - lr) * cos(yaw);
      y_check = y + ((2 * i - 1) * l / (2 * N) - lr) * sin(yaw);
      if (!isValid(x_check, y_check))
      {
        return false;
      }
    }
    return true;

    // return isValid(x, y);
  }

  bool rev_isValid(const ob::State *state)
  {
    if (!state)
    {
      throw std::runtime_error("No state found for vertex");
    }
    // ROS_INFO("rev_check");
    // Convert to RealVectorStateSpace
    const ob::RealVectorStateSpace::StateType *real_state =
        static_cast<const ob::RealVectorStateSpace::StateType *>(state);
    double x = real_state->values[0];
    double y = real_state->values[1];

    return isValid(x, y);
  }

  bool isValid(double x, double y)
  {
    if (!(x >= min_bound_x_ && x <= max_bound_x_ && y >= min_bound_y_ && y <= max_bound_y_))
    {
      // throw std::runtime_error("State must be within the bounds.");
      return false;
    }
    // find cell index in the map
    unsigned int idx = mapToCellIndex(x, y);
    auto occ = map_->data[idx];

    if (occ == 0)
      return true;
    return false;
  }

  bool smoother_isValid(double x, double y, double yaw)
  {
    int N = 3;
    double l = 4.7;
    // double w=2;
    double lr = 0.9;
    // double R = sqrt(pow(l/(2*N), 2) + pow(w/2, 2));

    double x_check = 0;
    double y_check = 0;

    for (int i = 0; i < N; ++i)
    {
      x_check = x + ((2 * i - 1) * l / (2 * N) - lr) * cos(yaw);
      y_check = y + ((2 * i - 1) * l / (2 * N) - lr) * sin(yaw);
      if (!isValid(x_check, y_check))
      {
        return false;
      }
    }
    return true;
  }

private:
  unsigned int mapToCellIndex(double x, double y)
  {
    // check if the state is within bounds
    if (!(x >= min_bound_x_ && x <= max_bound_x_ && y >= min_bound_y_ && y <= max_bound_y_))
    {
      // throw std::runtime_error("State must be within the bounds.");
      std::cout << "State must be within the bounds.";
    }

    int cell_x = (int)((x - map_origin_x_) / map_resolution_);
    int cell_y = (int)((y - map_origin_y_) / map_resolution_);

    unsigned int idx = cell_y * map_cell_size_x_ + cell_x;
    return idx;
  }

  inline bool getBounds(double &min_x, double &max_x, double &min_y, double &max_y, const nav_msgs::OccupancyGrid &ogm)
  {
    // extract map parameters
    unsigned int cells_size_x = ogm.info.width;
    unsigned int cells_size_y = ogm.info.height;
    double resolution = static_cast<double>(ogm.info.resolution);
    double origin_x = ogm.info.origin.position.x;
    double origin_y = ogm.info.origin.position.y;

    double map_size_x = cells_size_x * resolution;
    double map_size_y = cells_size_y * resolution;

    min_x = origin_x;
    min_y = origin_y;
    max_x = map_size_x - fabs(origin_x);
    max_y = map_size_y - fabs(origin_y);
    return true;
  }

  nav_msgs::OccupancyGridPtr map_;

  double min_bound_x_;
  double min_bound_y_;
  double max_bound_x_;
  double max_bound_y_;

  unsigned int map_cell_size_x_;
  unsigned int map_cell_size_y_;
  double map_resolution_;
  double map_origin_x_;
  double map_origin_y_;
};

// Geometry-only PCL LCCP. Binary float32 XYZ input -> uint32 label per input point.
// Fixed settings for the 2026-09-18 pilot; no fitted parameters or GT input.
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Expected input XYZ binary and output labels binary\n";
    return 1;
  }
  using Point = pcl::PointXYZRGBA;
  auto cloud = pcl::make_shared<pcl::PointCloud<Point>>();
  std::ifstream input(argv[1], std::ios::binary);
  float xyz[3];
  while (input.read(reinterpret_cast<char*>(xyz), sizeof(xyz))) {
    if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) ||
        !std::isfinite(xyz[2]) || xyz[2] <= 0) return 2;
    Point point;
    point.x = xyz[0]; point.y = xyz[1]; point.z = xyz[2];
    point.r = point.g = point.b = point.a = 255;
    cloud->push_back(point);
  }
  if (cloud->empty() || input.gcount() != 0) return 3;
  constexpr float voxel_m = .003f;  // Same physical voxel size as the localizer.
  constexpr float seed_m = .03f;    // PCL example default; no scene-specific fitting.
  pcl::SupervoxelClustering<Point> super(voxel_m, seed_m);
  super.setUseSingleCameraTransform(false);  // Keep metric XYZ, no depth warping.
  super.setInputCloud(cloud);
  super.setColorImportance(0.f);
  super.setSpatialImportance(1.f);
  super.setNormalImportance(4.f);
  std::map<std::uint32_t, pcl::Supervoxel<Point>::Ptr> patches;
  super.extract(patches);
  std::multimap<std::uint32_t, std::uint32_t> adjacency;
  super.getSupervoxelAdjacency(adjacency);
  pcl::LCCPSegmentation<Point> lccp;
  lccp.setConcavityToleranceThreshold(10.f);
  lccp.setSmoothnessCheck(true, voxel_m, seed_m, .1f);
  lccp.setSanityCheck(false);
  lccp.setKFactor(0);
  lccp.setMinSegmentSize(0);
  lccp.setInputSupervoxels(patches, adjacency);
  lccp.segment();
  auto labels = super.getLabeledCloud();
  lccp.relabelCloud(*labels);
  if (labels->size() != cloud->size()) return 4;
  std::ofstream output(argv[2], std::ios::binary);
  for (std::size_t i = 0; i < labels->size(); ++i) {
    const auto& q = labels->at(i);
    const auto& p = cloud->at(i);
    if (p.x != q.x || p.y != q.y || p.z != q.z) return 5;
    output.write(reinterpret_cast<const char*>(&q.label), sizeof(q.label));
  }
  if (!output) return 6;
  return 0;
}

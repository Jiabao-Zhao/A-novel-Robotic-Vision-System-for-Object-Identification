#!/usr/bin/env bash
# Isolated Ubuntu 22.04 packages; no sudo and no system/package-environment changes.
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"
package_dir="$project_dir/outputs/cache/lccp_packages"
prefix_dir="$project_dir/outputs/cache/lccp/prefix"
mkdir -p "$package_dir" "$prefix_dir" "$project_dir/outputs/cache/lccp"
cd "$package_dir"
apt-get download libpcl-dev libpcl-segmentation1.12 libpcl-features1.12 \
    libpcl-filters1.12 libpcl-search1.12 libpcl-kdtree1.12 libpcl-octree1.12 \
    libpcl-common1.12 libpcl-ml1.12 libpcl-sample-consensus1.12 \
    libboost1.74-dev libeigen3-dev libflann-dev
sha256sum ./*.deb > packages.sha256
for package in ./*.deb; do dpkg-deb -x "$package" "$prefix_dir"; done
g++ -std=c++17 -O2 -fopenmp \
    -I"$prefix_dir/usr/include/pcl-1.12" -I"$prefix_dir/usr/include/eigen3" \
    -I"$prefix_dir/usr/include" "$project_dir/scripts/lccp_segment.cpp" \
    -L"$prefix_dir/usr/lib/x86_64-linux-gnu" \
    -Wl,-rpath,"$prefix_dir/usr/lib/x86_64-linux-gnu" \
    -lpcl_segmentation -lpcl_features -lpcl_filters -lpcl_search -lpcl_kdtree \
    -lpcl_octree -lpcl_common -lpcl_sample_consensus \
    -o "$project_dir/outputs/cache/lccp/segment"

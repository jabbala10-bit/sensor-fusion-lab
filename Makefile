.PHONY: all cpp py lint clean

all: cpp py

cpp:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build cpp/build -j
	cd cpp && ./build/lidar_pipeline && ./build/fmcw_radar
	cd cpp && ([ -x build/camera_ttc ] && ./build/camera_ttc || echo "camera_ttc skipped: OpenCV 4 missing")
	cd cpp && ([ -x build/fusion_filters ] && ./build/fusion_filters || echo "fusion_filters skipped: Eigen missing")

py:
	cd python && python3 lidar_sim.py && python3 camera_sim.py && python3 radar_sim.py \
		&& python3 radar_mtt_sim.py && python3 kalman_sim.py

lint:
	ruff check .
	clang-format --dry-run --Werror $(shell find cpp -name '*.cpp')

clean:
	rm -rf cpp/build python/results python/__pycache__

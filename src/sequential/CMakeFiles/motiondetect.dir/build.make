# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/christian/School/ece408/final_project/src/sequential

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christian/School/ece408/final_project/src/sequential

# Include any dependencies generated for this target.
include CMakeFiles/motiondetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/motiondetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/motiondetect.dir/flags.make

CMakeFiles/motiondetect.dir/motiondetect.cpp.o: CMakeFiles/motiondetect.dir/flags.make
CMakeFiles/motiondetect.dir/motiondetect.cpp.o: motiondetect.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/christian/School/ece408/final_project/src/sequential/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/motiondetect.dir/motiondetect.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/motiondetect.dir/motiondetect.cpp.o -c /home/christian/School/ece408/final_project/src/sequential/motiondetect.cpp

CMakeFiles/motiondetect.dir/motiondetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motiondetect.dir/motiondetect.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/christian/School/ece408/final_project/src/sequential/motiondetect.cpp > CMakeFiles/motiondetect.dir/motiondetect.cpp.i

CMakeFiles/motiondetect.dir/motiondetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motiondetect.dir/motiondetect.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/christian/School/ece408/final_project/src/sequential/motiondetect.cpp -o CMakeFiles/motiondetect.dir/motiondetect.cpp.s

CMakeFiles/motiondetect.dir/motiondetect.cpp.o.requires:
.PHONY : CMakeFiles/motiondetect.dir/motiondetect.cpp.o.requires

CMakeFiles/motiondetect.dir/motiondetect.cpp.o.provides: CMakeFiles/motiondetect.dir/motiondetect.cpp.o.requires
	$(MAKE) -f CMakeFiles/motiondetect.dir/build.make CMakeFiles/motiondetect.dir/motiondetect.cpp.o.provides.build
.PHONY : CMakeFiles/motiondetect.dir/motiondetect.cpp.o.provides

CMakeFiles/motiondetect.dir/motiondetect.cpp.o.provides.build: CMakeFiles/motiondetect.dir/motiondetect.cpp.o

# Object files for target motiondetect
motiondetect_OBJECTS = \
"CMakeFiles/motiondetect.dir/motiondetect.cpp.o"

# External object files for target motiondetect
motiondetect_EXTERNAL_OBJECTS =

motiondetect: CMakeFiles/motiondetect.dir/motiondetect.cpp.o
motiondetect: CMakeFiles/motiondetect.dir/build.make
motiondetect: /usr/lib/libopencv_videostab.so.2.4.10
motiondetect: /usr/lib/libopencv_video.so.2.4.10
motiondetect: /usr/lib/libopencv_ts.a
motiondetect: /usr/lib/libopencv_superres.so.2.4.10
motiondetect: /usr/lib/libopencv_stitching.so.2.4.10
motiondetect: /usr/lib/libopencv_photo.so.2.4.10
motiondetect: /usr/lib/libopencv_ocl.so.2.4.10
motiondetect: /usr/lib/libopencv_objdetect.so.2.4.10
motiondetect: /usr/lib/libopencv_nonfree.so.2.4.10
motiondetect: /usr/lib/libopencv_ml.so.2.4.10
motiondetect: /usr/lib/libopencv_legacy.so.2.4.10
motiondetect: /usr/lib/libopencv_imgproc.so.2.4.10
motiondetect: /usr/lib/libopencv_highgui.so.2.4.10
motiondetect: /usr/lib/libopencv_gpu.so.2.4.10
motiondetect: /usr/lib/libopencv_flann.so.2.4.10
motiondetect: /usr/lib/libopencv_features2d.so.2.4.10
motiondetect: /usr/lib/libopencv_core.so.2.4.10
motiondetect: /usr/lib/libopencv_contrib.so.2.4.10
motiondetect: /usr/lib/libopencv_calib3d.so.2.4.10
motiondetect: /lib64/libGLU.so
motiondetect: /lib64/libGL.so
motiondetect: /lib64/libSM.so
motiondetect: /lib64/libICE.so
motiondetect: /lib64/libX11.so
motiondetect: /lib64/libXext.so
motiondetect: /usr/lib/libopencv_nonfree.so.2.4.10
motiondetect: /usr/lib/libopencv_ocl.so.2.4.10
motiondetect: /usr/lib/libopencv_gpu.so.2.4.10
motiondetect: /usr/lib/libopencv_photo.so.2.4.10
motiondetect: /usr/lib/libopencv_objdetect.so.2.4.10
motiondetect: /usr/lib/libopencv_legacy.so.2.4.10
motiondetect: /usr/lib/libopencv_video.so.2.4.10
motiondetect: /usr/lib/libopencv_ml.so.2.4.10
motiondetect: /usr/lib/libopencv_calib3d.so.2.4.10
motiondetect: /usr/lib/libopencv_features2d.so.2.4.10
motiondetect: /usr/lib/libopencv_highgui.so.2.4.10
motiondetect: /usr/lib/libopencv_imgproc.so.2.4.10
motiondetect: /usr/lib/libopencv_flann.so.2.4.10
motiondetect: /usr/lib/libopencv_core.so.2.4.10
motiondetect: CMakeFiles/motiondetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable motiondetect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/motiondetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/motiondetect.dir/build: motiondetect
.PHONY : CMakeFiles/motiondetect.dir/build

CMakeFiles/motiondetect.dir/requires: CMakeFiles/motiondetect.dir/motiondetect.cpp.o.requires
.PHONY : CMakeFiles/motiondetect.dir/requires

CMakeFiles/motiondetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/motiondetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/motiondetect.dir/clean

CMakeFiles/motiondetect.dir/depend:
	cd /home/christian/School/ece408/final_project/src/sequential && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christian/School/ece408/final_project/src/sequential /home/christian/School/ece408/final_project/src/sequential /home/christian/School/ece408/final_project/src/sequential /home/christian/School/ece408/final_project/src/sequential /home/christian/School/ece408/final_project/src/sequential/CMakeFiles/motiondetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/motiondetect.dir/depend

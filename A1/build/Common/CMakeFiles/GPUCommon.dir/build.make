# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/stud/s_wodtke/Downloads/Assignment1/A1/Assignment1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stud/s_wodtke/Downloads/Assignment1/A1/build

# Include any dependencies generated for this target.
include Common/CMakeFiles/GPUCommon.dir/depend.make

# Include the progress variables for this target.
include Common/CMakeFiles/GPUCommon.dir/progress.make

# Include the compile flags for this target's objects.
include Common/CMakeFiles/GPUCommon.dir/flags.make

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o: Common/CMakeFiles/GPUCommon.dir/flags.make
Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o: /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CAssignmentBase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stud/s_wodtke/Downloads/Assignment1/A1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o -c /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CAssignmentBase.cpp

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.i"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CAssignmentBase.cpp > CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.i

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.s"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CAssignmentBase.cpp -o CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.s

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.requires:

.PHONY : Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.requires

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.provides: Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/GPUCommon.dir/build.make Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.provides.build
.PHONY : Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.provides

Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.provides.build: Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o


Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o: Common/CMakeFiles/GPUCommon.dir/flags.make
Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o: /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CTimer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stud/s_wodtke/Downloads/Assignment1/A1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/GPUCommon.dir/CTimer.cpp.o -c /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CTimer.cpp

Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GPUCommon.dir/CTimer.cpp.i"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CTimer.cpp > CMakeFiles/GPUCommon.dir/CTimer.cpp.i

Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GPUCommon.dir/CTimer.cpp.s"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CTimer.cpp -o CMakeFiles/GPUCommon.dir/CTimer.cpp.s

Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.requires:

.PHONY : Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.requires

Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.provides: Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/GPUCommon.dir/build.make Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.provides.build
.PHONY : Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.provides

Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.provides.build: Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o


Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o: Common/CMakeFiles/GPUCommon.dir/flags.make
Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o: /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CLUtil.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stud/s_wodtke/Downloads/Assignment1/A1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/GPUCommon.dir/CLUtil.cpp.o -c /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CLUtil.cpp

Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GPUCommon.dir/CLUtil.cpp.i"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CLUtil.cpp > CMakeFiles/GPUCommon.dir/CLUtil.cpp.i

Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GPUCommon.dir/CLUtil.cpp.s"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/stud/s_wodtke/Downloads/Assignment1/A1/Common/CLUtil.cpp -o CMakeFiles/GPUCommon.dir/CLUtil.cpp.s

Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.requires:

.PHONY : Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.requires

Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.provides: Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/GPUCommon.dir/build.make Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.provides.build
.PHONY : Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.provides

Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.provides.build: Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o


# Object files for target GPUCommon
GPUCommon_OBJECTS = \
"CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o" \
"CMakeFiles/GPUCommon.dir/CTimer.cpp.o" \
"CMakeFiles/GPUCommon.dir/CLUtil.cpp.o"

# External object files for target GPUCommon
GPUCommon_EXTERNAL_OBJECTS =

Common/libGPUCommon.a: Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o
Common/libGPUCommon.a: Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o
Common/libGPUCommon.a: Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o
Common/libGPUCommon.a: Common/CMakeFiles/GPUCommon.dir/build.make
Common/libGPUCommon.a: Common/CMakeFiles/GPUCommon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/stud/s_wodtke/Downloads/Assignment1/A1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libGPUCommon.a"
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && $(CMAKE_COMMAND) -P CMakeFiles/GPUCommon.dir/cmake_clean_target.cmake
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GPUCommon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Common/CMakeFiles/GPUCommon.dir/build: Common/libGPUCommon.a

.PHONY : Common/CMakeFiles/GPUCommon.dir/build

Common/CMakeFiles/GPUCommon.dir/requires: Common/CMakeFiles/GPUCommon.dir/CAssignmentBase.cpp.o.requires
Common/CMakeFiles/GPUCommon.dir/requires: Common/CMakeFiles/GPUCommon.dir/CTimer.cpp.o.requires
Common/CMakeFiles/GPUCommon.dir/requires: Common/CMakeFiles/GPUCommon.dir/CLUtil.cpp.o.requires

.PHONY : Common/CMakeFiles/GPUCommon.dir/requires

Common/CMakeFiles/GPUCommon.dir/clean:
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common && $(CMAKE_COMMAND) -P CMakeFiles/GPUCommon.dir/cmake_clean.cmake
.PHONY : Common/CMakeFiles/GPUCommon.dir/clean

Common/CMakeFiles/GPUCommon.dir/depend:
	cd /home/stud/s_wodtke/Downloads/Assignment1/A1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stud/s_wodtke/Downloads/Assignment1/A1/Assignment1 /home/stud/s_wodtke/Downloads/Assignment1/A1/Common /home/stud/s_wodtke/Downloads/Assignment1/A1/build /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common /home/stud/s_wodtke/Downloads/Assignment1/A1/build/Common/CMakeFiles/GPUCommon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Common/CMakeFiles/GPUCommon.dir/depend


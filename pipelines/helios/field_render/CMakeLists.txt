# field_render, a Helios project. Configure with -DHELIOS_DIR=<Helios checkout> (absolute path);
# pipelines/scripts/image_simulation/build_helios.sh does this.
cmake_minimum_required(VERSION 3.15)
project(helios)
set( BASE_DIRECTORY "${HELIOS_DIR}" )
set( EXECUTABLE_NAME "field_render" )
set( SOURCE_FILES "main.cpp" )
set( PLUGINS "radiation;plantarchitecture;leafoptics" )
include( "${BASE_DIRECTORY}/core/CMake_project.cmake" )

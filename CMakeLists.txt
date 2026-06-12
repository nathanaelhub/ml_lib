cmake_minimum_required(VERSION 3.15)
project(ml_lib LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- library ----
add_library(ml_lib STATIC ml_lib.c)
target_include_directories(ml_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(ml_lib PRIVATE /W4)
else()
    target_compile_options(ml_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(ml_lib PUBLIC m)          # libm for exp/tanh/sqrt
endif()

# ---- demo executable ----
add_executable(ml_demo main.c)
target_link_libraries(ml_demo PRIVATE ml_lib)

# ---- ctest: the demo self-checks (OR / XOR / gradient check) and
#      returns non-zero on any failure, so it doubles as a smoke test ----
enable_testing()
add_test(NAME ml_demo COMMAND ml_demo)
# Run from the source tree so the demo finds data/xor2d.csv.
set_tests_properties(ml_demo PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_minimum_required(VERSION 3.16)
project(cpp-llamalib LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Library target (header-only wrapper around llama.cpp)
# ---------------------------------------------------------------------------
macro(create_cpp_llamalib_target)
  add_library(cpp-llamalib INTERFACE)
  target_include_directories(cpp-llamalib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
  target_link_libraries(cpp-llamalib INTERFACE llama)
endmacro()

# When used as a subproject, the parent must provide the llama target
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  create_cpp_llamalib_target()
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default test model: Gemma 3 1B IT Q4_K_M (~806MB)
set(DEFAULT_MODEL_NAME "gemma-3-1b-it-Q4_K_M.gguf")
set(DEFAULT_MODEL_URL
  "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/${DEFAULT_MODEL_NAME}")
set(DEFAULT_MODEL_DIR "${CMAKE_BINARY_DIR}/models")
set(DEFAULT_MODEL_PATH "${DEFAULT_MODEL_DIR}/${DEFAULT_MODEL_NAME}")

if(NOT DEFINED MODEL_PATH)
  if(NOT EXISTS "${DEFAULT_MODEL_PATH}")
    message(STATUS "Downloading default test model: ${DEFAULT_MODEL_NAME} ...")
    file(MAKE_DIRECTORY "${DEFAULT_MODEL_DIR}")
    file(DOWNLOAD
      "${DEFAULT_MODEL_URL}"
      "${DEFAULT_MODEL_PATH}"
      SHOW_PROGRESS
      STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
      list(GET download_status 1 error_msg)
      message(FATAL_ERROR "Failed to download model: ${error_msg}")
    endif()
  else()
    message(STATUS "Using cached model: ${DEFAULT_MODEL_PATH}")
  endif()
  set(MODEL_PATH "${DEFAULT_MODEL_PATH}")
endif()

include(FetchContent)

# Fetch llama.cpp
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        b8851
  GIT_SHALLOW    TRUE
)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama_cpp)

create_cpp_llamalib_target()

# Fetch Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(tests tests/test_generate.cpp)

target_link_libraries(tests PRIVATE
  Catch2::Catch2WithMain
  cpp-llamalib
)

target_compile_definitions(tests PRIVATE
  MODEL_PATH="${MODEL_PATH}"
)

# Register tests with CTest
option(BUILD_TESTING "Enable test discovery" ON)
if(BUILD_TESTING)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)
  catch_discover_tests(tests)
endif()

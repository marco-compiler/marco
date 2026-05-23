include_guard()
include(LLVMDistributionSupport)

macro(set_marco_windows_version_resource_properties name)
  if (DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file}
      VERSION_MAJOR ${MARCO_VERSION_MAJOR}
      VERSION_MINOR ${MARCO_VERSION_MINOR}
      VERSION_PATCHLEVEL ${MARCO_VERSION_PATCHLEVEL}
      VERSION_STRING "${MARCO_VERSION}"
      PRODUCT_NAME "marco")
  endif()
endmacro()

# Convert the "friendly" name of a variable into its target name.
# For example: utils -> MARCOUtils
function(marco_canonicalize_library_name canonical_name name)
  # Get first letter and capitalize.
  string(SUBSTRING ${name} 0 1 first-letter)
  string(TOUPPER ${first-letter} first-letter)

  # Get the rest of the name.
  string(LENGTH ${name} length)
  execute_process(COMMAND expr ${length} - 1 OUTPUT_VARIABLE length)
  string(SUBSTRING ${name} 1 ${length} rest)

  # Compose macro-name with first letter uppercase.
  set(${canonical_name} "MARCO${first-letter}${rest}" PARENT_SCOPE)
endfunction()

# Declare a MARCO library.
macro(marco_add_library name)
  cmake_parse_arguments(ARG
    ""
    ""
    "CLANG_LIBS MLIR_LIBS"
    ${ARGN})

  marco_canonicalize_library_name(canonical_name ${name})
  set_property(GLOBAL APPEND PROPERTY MARCO_LIBS ${canonical_name})

  add_mlir_library(${canonical_name} OUTPUT_NAME ${canonical_name} ${MARCO_LIB_TYPE} ${ARG_UNPARSED_ARGUMENTS})
  add_library(marco::${name} ALIAS ${canonical_name})

  if (ARG_CLANG_LIBS)
    clang_target_link_libraries(${canonical_name} PRIVATE ${ARG_CLANG_LIBS})
  endif()

  if (ARG_MLIR_LIBS)
    mlir_target_link_libraries(${canonical_name} PRIVATE ${ARG_MLIR_LIBS})
  endif()

  mlir_check_all_link_libraries(${canonical_name})
  set_marco_windows_version_resource_properties(${canonical_name})

  #install(TARGETS ${name}
  #    COMPONENT ${name}
  #    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  #    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endmacro()

# Declare a MARCO executable.
macro(marco_add_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "CLANG_LIBS MLIR_LIBS"
    ${ARGN})

  add_llvm_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})

  if (ARG_CLANG_LIBS)
    clang_target_link_libraries(${name} PRIVATE ${ARG_CLANG_LIBS})
  endif()

  if (ARG_MLIR_LIBS)
    mlir_target_link_libraries(${name} PRIVATE ${ARG_MLIR_LIBS})
  endif()

  mlir_check_all_link_libraries(${name})
  set_marco_windows_version_resource_properties(${name})
endmacro()

# Declare a MARCO tool.
macro(marco_add_tool name)
  if (NOT MARCO_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  marco_add_executable(${name} ${ARGN})
  add_dependencies(MARCO-Tools ${name})

  if (MARCO_BUILD_TOOLS)
    get_target_export_arg(${name} MARCO export_to_marcotargets)

    install(TARGETS ${name}
      ${export_to_marcotargets}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT ${name})

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT ${name})
    endif()
    
    set_property(GLOBAL APPEND PROPERTY MARCO_EXPORTS ${name})
  endif()
endmacro()

# Declare a symlink to a MARCO tool.
macro(marco_add_symlink name dest)
  llvm_add_tool_symlink(MARCO ${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(MARCO ${name} ${dest} ALWAYS_GENERATE)
endmacro()

# Declare a MARCO unit test leveraging Google Test
function(marco_add_unittest test_name)
  set(test_suite MARCOUnitTests)
  add_llvm_executable(${test_name} ${ARGN})

  target_link_directories(${test_name} PRIVATE ${MARCO_LIBS_DIR})
  target_link_libraries(${test_name} PRIVATE gtest_main gmock)

  add_dependencies(${test_suite} ${test_name})
  get_target_property(test_suite_folder ${test_suite} FOLDER)

  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif()

  gtest_discover_tests(${test_name})
endfunction()

# Convert the "friendly" names of MARCO libraries into the ones to be used for linking
function(marco_map_components_to_libnames out_libs)
  set(link_components ${ARGN})

  if(NOT MARCO_AVAILABLE_LIBS)
    # Inside MARCO itself available libs are in a global property.
    get_property(MARCO_AVAILABLE_LIBS GLOBAL PROPERTY MARCO_LIBS)
  endif()

  string(TOUPPER "${MARCO_AVAILABLE_LIBS}" capitalized_libs)

  foreach(c ${link_components})
    get_property(c_rename GLOBAL PROPERTY LLVM_COMPONENT_NAME_${c})

    if(c_rename)
      set(c ${c_rename})
    endif()

    # Canonize the component name
    marco_canonize_library_name(canonized_name ${c})
    string(TOUPPER "${canonized_name}" capitalized)
    list(FIND capitalized_libs ${capitalized} lib_idx)

    if(lib_idx LESS 0)
      marco_canonize_library_name(canonical_name ${c})
      list(APPEND expanded_components ${canonical_name})
    else()
      list(GET MARCO_AVAILABLE_LIBS ${lib_idx} canonical_lib)
      list(APPEND expanded_components ${canonical_lib})
    endif()
  endforeach(c)

  set(${out_libs} ${expanded_components} PARENT_SCOPE)
endfunction()

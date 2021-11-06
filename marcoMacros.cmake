# Convert the "friendly" name of a variable into its target name.
# For example: utils -> MARCOUtils
function(marco_canonize_library_name canonical_name name)
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

# Declare a MARCO library
macro(marco_add_library name)
	marco_canonize_library_name(canonized_name ${name})
	set_property(GLOBAL APPEND PROPERTY MARCO_LIBS ${canonized_name})

	add_library(${name} ${ARGN})
	add_library(marco::${name} ALIAS ${name})

	set_target_properties(${name} PROPERTIES OUTPUT_NAME ${canonized_name})
	target_compile_features(${name} PUBLIC cxx_std_17)

	target_include_directories(${name} PRIVATE
			src
			PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>)

	include(GNUInstallDirs)
	install(TARGETS ${name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
	install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endmacro()

# Declare a MARCO tool
macro(marco_add_tool name)
	add_llvm_executable(${name} ${ARGN})
	add_executable(marco::${name} ALIAS ${name})

	llvm_update_compile_flags(${name})

	target_compile_features(${name} PUBLIC cxx_std_17)

	include(GNUInstallDirs)
	install(TARGETS ${name} RUNTIME DESTINATION bin)
endmacro()

# Declare a MARCO unit test leveraging Google Test
macro(marco_add_unittest test_suite test_name)
	add_executable(${test_name} ${ARGN})
	target_compile_features(${test_name} PUBLIC cxx_std_17)

	include(GoogleTest)
	target_link_libraries(${test_name} PRIVATE gmock gtest_main)
	gtest_discover_tests(${test_name})
endmacro()

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

###
###  marco_link_llvm_libs  
###
###  Same as llvm_config, but automatically turns on shared libs if the
###  LLVM we are linking with has them available. Everything is linked as
###  PUBLIC so that everything works with the way we use CMake.
###    "Shouldn't have this been done by LLVM's Cmake stuff already?" You bet!
###  "Does it do that then?" NO!!
###
###             ----->>>> DO NOT REMOVE THIS FUNCTION! <<<<----
###
###   EVEN IF IT SEEMS NOT NECESSARY ON ***YOUR*** MACHINE IT DOES NOT MEAN
###                     IT IS USELESS FOR EVERYBODY
###
###    ----->>>> NEVER USE llvm_config, USE THIS FUNCTION INSTEAD <<<<-----
###
###     IF YOU DON'T, THE BUILD WILL BREAK FOR EVERYBODY WHO HAS BOTH THE
###                STATIC AND DYNAMIC LLVM LIBRARIES AVAILABLE
###
function(marco_link_llvm_libs target)
  set(link_components ${ARGN})

	if(LLVM IN_LIST LLVM_AVAILABLE_LIBS)
	  if (DEFINED link_components AND DEFINED LLVM_DYLIB_COMPONENTS)
      if("${LLVM_DYLIB_COMPONENTS}" STREQUAL "all")
        set(link_components "")
      else()
        list(REMOVE_ITEM link_components ${LLVM_DYLIB_COMPONENTS})
      endif()
    endif()

    target_link_libraries(${target} PUBLIC LLVM)
	endif()
	
	llvm_map_components_to_libnames(libs ${link_components})
	target_link_libraries(${target} PUBLIC ${libs})
endfunction(marco_link_llvm_libs)

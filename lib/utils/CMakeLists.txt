ADD_LIBRARY(utils SHARED src/Utils.cpp src/Lexer.cpp)
ADD_LIBRARY(modelica::utils ALIAS utils)
TARGET_INCLUDE_DIRECTORIES(utils 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(utils PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_utils_libs Support)

TARGET_LINK_LIBRARIES(utils PUBLIC ${llvm_utils_libs})

modelicaInstall(utils)

ADD_SUBDIRECTORY(test)

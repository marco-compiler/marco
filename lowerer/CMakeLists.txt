ADD_LIBRARY(lowerer STATIC src/SimExp.cpp src/Simulation.cpp src/SimLowerer.cpp src/SimErrors.cpp src/SimType.cpp) 
ADD_LIBRARY(modelica::lowerer ALIAS lowerer)
TARGET_INCLUDE_DIRECTORIES(lowerer 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(lowerer PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_lowerer_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(lowerer PUBLIC ${llvm_lowerer_libs} modelica::utils)

modelicaInstall(lowerer)

add_subdirectory(test)

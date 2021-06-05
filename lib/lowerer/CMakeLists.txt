set(SOURCES
		src/Lowerer.cpp
		src/LowererUtils.cpp
		src/ExpLowerer.cpp
		src/BltBlockLowerer.cpp
		src/CallLowerer.cpp
		src/OperationLowerer.cpp)
		src/IdaSolver.cpp

marco_add_library(lowerer SHARED ${SOURCES})

marco_link_llvm_libs(lowerer USE_SHARED bitwriter core support)

target_link_libraries(lowerer
		PUBLIC marco::passes
		PUBLIC marco::utils
		PRIVATE sundials)

add_subdirectory(test)

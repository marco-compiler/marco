set(SOURCES
		src/PassManager.cpp
		src/SolveModel.cpp
		src/ForwardEuler.cpp)

marco_add_library(passes SHARED ${SOURCES})

llvm_config(passes USE_SHARED core support)

target_link_libraries(passes
		PUBLIC marco::matching)

add_subdirectory(test)

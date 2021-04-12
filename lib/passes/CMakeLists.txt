set(SOURCES
		src/PassManager.cpp
		src/SolveModel.cpp
		src/ForwardEuler.cpp
		src/BruteDAE.cpp)

marco_add_library(passes SHARED ${SOURCES})

marco_link_llvm_libs(passes USE_SHARED core support)

target_link_libraries(passes
		PUBLIC marco::matching)

add_subdirectory(test)

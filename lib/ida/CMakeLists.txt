set(SOURCES
		src/IdaSolver.cpp)

marco_add_library(ida SHARED ${SOURCES})

llvm_config(ida USE_SHARED core support)

target_link_libraries(ida
		PRIVATE marco::mlirlowerer
		PRIVATE marco::runtime
		PRIVATE marco::utils
		PRIVATE sundials)

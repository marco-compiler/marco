set(SOURCES
		src/BuiltInFunctions.cpp
		src/PrintFunctions.cpp)

marco_add_library(runtime SHARED ${SOURCES})

llvm_config(runtime USE_SHARED support)

add_subdirectory(test)

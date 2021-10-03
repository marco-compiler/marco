set(SOURCES
		src/Utils.cpp
		src/Lexer.cpp
		src/Interval.cpp
		src/IndexSet.cpp
		src/LogMessage.cpp
		src/SourcePosition.cpp
		src/ThreadPool.cpp
		src/VariableFilter.cpp)

marco_add_library(utils SHARED ${SOURCES})

llvm_config(utils USE_SHARED core support)

add_subdirectory(test)

set(SOURCES
		src/Utils.cpp
		src/Lexer.cpp
		src/Interval.cpp
        src/Range.cpp
		src/LogMessage.cpp
		src/SourcePosition.cpp
		src/ThreadPool.cpp
		src/VariableFilter.cpp)

marco_add_library(utils SHARED ${SOURCES})

marco_link_llvm_libs(utils USE_SHARED core support)

add_subdirectory(test)

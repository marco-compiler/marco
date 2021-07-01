marcoAddLibrary(
		utils SHARED
		src/Utils.cpp
		src/Lexer.cpp
		src/Interval.cpp
		src/IndexSet.cpp
		src/SourcePosition.cpp
		src/ThreadPool.cpp
		src/VariableFilter.cpp
		src/VariableFilterParser.cpp
		src/VariableTracker.cpp)

marco_link_llvm_libs(utils support)

modelicaAddLibrary(
		utils SHARED
		src/Utils.cpp
		src/Lexer.cpp
		src/Interval.cpp
		src/IndexSet.cpp
		src/SourceRange.cpp
		src/ThreadPool.cpp)

modelica_link_llvm_libs(utils support)

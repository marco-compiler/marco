modelicaAddLibrary(
		utils SHARED
		src/Utils.cpp
		src/Lexer.cpp
		src/Interval.cpp
		src/IndexSet.cpp
		src/SourceRange.cpp
		src/ThreadPool.cpp)

llvm_map_components_to_libnames(llvm_utils_libs Support)
TARGET_LINK_LIBRARIES(utils PUBLIC ${llvm_utils_libs})

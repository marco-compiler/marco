marcoAddLibrary(
		runtime SHARED
		src/BuiltInFunctions.cpp
		src/PrintFunctions.cpp)

marco_link_llvm_libs(runtime support)

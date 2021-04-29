modelicaAddLibrary(
		runtime SHARED
		src/BuiltInFunctions.cpp
		src/PrintFunctions.cpp)

modelica_link_llvm_libs(runtime support)

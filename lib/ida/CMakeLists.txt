marcoAddLibrary(
	ida SHARED
	src/IdaSolver.cpp
)

target_link_libraries(
	ida PUBLIC
	marco::mlirlowerer
	marco::runtime
	marco::utils
	sundials
)

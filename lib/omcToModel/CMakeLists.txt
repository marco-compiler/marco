marcoAddLibrary(
	omcToModel SHARED 
	src/OmcToModelPass.cpp) 

target_link_libraries(omcToModel PUBLIC marco::frontend)


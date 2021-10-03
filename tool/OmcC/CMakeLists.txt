set(SOURCES
		src/Main.cpp)

marco_add_tool(omcc ${SOURCES})

target_link_libraries(omcc
		PRIVATE marco::omcToModel
		PRIVATE marco::lowerer)

add_subdirectory(test)

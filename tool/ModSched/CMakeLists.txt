set(SOURCES
		src/Main.cpp)

marco_add_tool(modsched ${SOURCES})

target_link_libraries(modsched
		PRIVATE marco::model
		PRIVATE marco::passes)

add_subdirectory(test)

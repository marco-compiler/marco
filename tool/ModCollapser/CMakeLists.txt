set(SOURCES
		src/Main.cpp)

marco_add_tool(modcollapser ${SOURCES})

target_link_libraries(modcollapser
		PRIVATE marco::model
		PRIVATE marco::passes)

add_subdirectory(test)

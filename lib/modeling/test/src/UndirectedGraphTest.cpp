#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <llvm/ADT/StringRef.h>
#include <marco/modeling/Graph.h>

using namespace marco::modeling::internal;
using namespace testing;

class Vertex
{
  public:
  Vertex(llvm::StringRef name, int value = 0) : name(name.str()), value(value)
  {
  }

  bool operator==(const Vertex& other) const
  {
    return name == other.name;
  }

  llvm::StringRef getName() const
  {
    return name;
  }

  int getValue() const
  {
    return value;
  }

  private:
  std::string name;
  int value;
};

class Edge
{
  public:
  Edge(llvm::StringRef name, int value = 0) : name(name.str()), value(value)
  {
  }

  bool operator==(const Edge& other) const
  {
    return name == other.name;
  }

  llvm::StringRef getName() const
  {
    return name;
  }

  int getValue() const
  {
    return value;
  }

  private:
  std::string name;
  int value;
};

TEST(UndirectedGraph, addVertex)
{
  UndirectedGraph<Vertex, Edge> graph;
  auto x = graph.addVertex(Vertex("x"));
  EXPECT_EQ(graph[x].getName(), "x");
}

TEST(UndirectedGraph, filteredVertices)
{
  UndirectedGraph<Vertex, Edge> graph;

  Vertex x("x", 1);
  Vertex y("y", 0);
  Vertex z("z", 1);

  graph.addVertex(x);
  graph.addVertex(y);
  graph.addVertex(z);

  auto filter = [](const Vertex& vertex) -> bool {
      return vertex.getValue() == 1;
  };

  std::vector<Vertex> vertices;

  for (const auto& vertexDescriptor : graph.getVertices(filter))
    vertices.push_back(graph[vertexDescriptor]);

  EXPECT_THAT(vertices, UnorderedElementsAre(x, z));
}

TEST(UndirectedGraph, addEdge)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  EXPECT_EQ(graph[e1].getName(), "e1");
}

TEST(UndirectedGraph, incidentEdges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);

  Edge e3("e3");
  graph.addEdge(y, z, e3);

  std::vector<Edge> edges;

  for (const auto& edgeDescriptor : graph.getOutgoingEdges(x))
    edges.push_back(graph[edgeDescriptor]);

  EXPECT_THAT(edges, UnorderedElementsAre(e1, e2));
}

TEST(UndirectedGraph, edges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);

  Edge e3("e3");
  graph.addEdge(y, z, e3);

  std::vector<Edge> edges;

  for (const auto& edgeDescriptor : graph.getEdges())
    edges.push_back(graph[edgeDescriptor]);

  EXPECT_THAT(edges, UnorderedElementsAre(e1, e2, e3));
}

TEST(UndirectedGraph, filteredEdges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1", 1);
  Edge e2("e2", 0);

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);

  Edge e3("e3", 1);
  graph.addEdge(y, z, e3);

  auto filter = [](const Edge& edge) -> bool {
      return edge.getValue() == 1;
  };

  std::vector<Edge> edges;

  for (const auto& edgeDescriptor : graph.getEdges(filter))
    edges.push_back(graph[edgeDescriptor]);

  EXPECT_THAT(edges, UnorderedElementsAre(e1, e3));
}

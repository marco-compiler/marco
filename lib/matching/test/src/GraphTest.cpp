#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
#include <marco/matching/Graph.h>

using namespace marco::matching;

class Vertex
{
  public:
  Vertex(llvm::StringRef name) : name(name.str())
  {
  }

  llvm::StringRef getName() const
  {
    return name;
  }

  private:
  std::string name;
};

class Edge
{
  public:
  Edge(llvm::StringRef name) : name(name.str())
  {
  }

  llvm::StringRef getName() const
  {
    return name;
  }

  private:
  std::string name;
};

TEST(Matching, graphAddVertex)
{
  Graph<Vertex, Edge> graph;
  auto x = graph.addVertex(Vertex("x"));
  EXPECT_EQ(graph[x].getName(), "x");
}

TEST(Matching, graphAddEdge)
{
  Graph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  EXPECT_EQ(graph[e1].getName(), "e1");
}

TEST(Matching, graphIncidentEdge)
{
  Graph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  //auto edges = graph.getIncidentEdges(x);

}